import copy
import json
import os
import warnings
from absl import app, flags

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10, FashionMNIST
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange

from score.both import get_inception_and_fid_score

from model import DenoisingNet

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')

# Denoising Diffusion Network
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_integer('tau', 100, help='DDIM sampling steps')
flags.DEFINE_string('net_cfg', './config/net/unet_small.json', help='path of network configuration')

# Dataset
flags.DEFINE_string('dataset', 'CIFAR10', help='dataset selection (CIFAR10/FashionMNIST)')

# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='size of image')
flags.DEFINE_integer('img_ch', 3, help='image channel')
flags.DEFINE_integer('warmup', 200, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 8, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate (non-positive means disable ema)")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
flags.DEFINE_string('resume_from_ckpt', '', help='continue training from certain checkpoint')
flags.DEFINE_bool('resume_without_progress', False, help='load checkpoint but ignore the saved progress')
flags.DEFINE_spaceseplist('resume_layer_excluded', '', help='layer not loaded when resume_from_ckpt is set')

flags.DEFINE_float('layer_loss_factor', 1., help='the factor multiplied on 1/alphas_bar_t for layer loss weight gamma_t')

# Logging & Sampling
flags.DEFINE_string('logdir', './ckpts', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 100, help='frequency of sampling')

# Evaluation
flags.DEFINE_integer('save_step', 500, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 20000, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', True, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache (must match the train set)')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key, param in source.named_parameters():
        if not param.requires_grad: continue
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))
    return


def infiniteloop(dataloader):
    epoch_cnt = 0
    while True:
        for x, _ in iter(dataloader):
            yield x, epoch_cnt
        epoch_cnt += 1


def warmup_lr(sched_step):
    return min(sched_step, FLAGS.warmup) / FLAGS.warmup


def _eval(model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = model.get_noise(batch_size, device)
            batch_images = model(x_T)[:, 0, ...].clip(-1, 1).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(images, FLAGS.fid_cache, num_images=FLAGS.num_images,
                                                    use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def train():
    # dataset setup
    dataset_norm_factor = tuple([0.5]*FLAGS.img_ch)
    dataset_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.Resize(FLAGS.img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(dataset_norm_factor, dataset_norm_factor)])
    assert FLAGS.dataset in ['FashionMNIST', 'CIFAR10']
    dataset = eval(FLAGS.dataset)(root='./data', train=True, download=True, transform=dataset_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True,
                                             num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # model setup
    with open(FLAGS.net_cfg) as f:
        model_cfg = json.loads(f.read())
    net_model = DenoisingNet(T=FLAGS.T, tau=FLAGS.tau, **model_cfg,
                             img_size=FLAGS.img_size, img_ch=FLAGS.img_ch,
                             beta_1=FLAGS.beta_1, beta_T=FLAGS.beta_T).to(device)
    ema_model = copy.deepcopy(net_model) if FLAGS.ema_decay > 0 else None
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        if ema_model is not None:
            ema_model = torch.nn.DataParallel(ema_model)
        net_model = torch.nn.DataParallel(net_model)
    sample_noise = net_model.get_noise(FLAGS.sample_size, device)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        if not param.requires_grad: continue
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # load checkpoint
    start_step, base_epoch = -1, 0
    if len(FLAGS.resume_from_ckpt) != 0:
        layer_excluded = list(map(int, FLAGS.resume_layer_excluded))
        ckpt = torch.load(FLAGS.resume_from_ckpt, map_location=device)
        net_model.load_checkpoint(ckpt['net_model'], layer_excluded)
        if ema_model is not None:
            if ckpt['ema_model'] is None: ema_model = None
            else: ema_model.load_checkpoint(ckpt['ema_model'], layer_excluded)
        if not FLAGS.resume_without_progress:
            sched.load_state_dict(ckpt['sched'])
            optim.load_state_dict(ckpt['optim'])
            start_step = ckpt['step']
            base_epoch = ckpt['epoch']
            sample_noise = ckpt['sample_noise']

    # log setup
    os.makedirs(FLAGS.logdir, exist_ok=True)
    purge_step = start_step if len(FLAGS.resume_from_ckpt) != 0 else None
    writer = SummaryWriter(FLAGS.logdir, purge_step=purge_step)
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'), exist_ok=True)
    if len(FLAGS.resume_from_ckpt) == 0:
        grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
        writer.add_image('real_sample', grid)
        writer.flush()

    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())

    # start training
    with trange(start_step+1, FLAGS.total_steps, dynamic_ncols=True) as pbar:
        pbar_postfix = {'epoch': 'N/A', 'loss': 'N/A', 'lr': 'N/A'}
        # Calculate coefficient for intermediate losses
        loss_coef = 1.0 / net_model.get_alphas_bar()
        loss_coef = FLAGS.layer_loss_factor*loss_coef.to(device)

        for step in pbar:
            # train
            # Get new mini-batch
            x_0, new_epoch = next(datalooper)
            x_0 = x_0.to(device)
            epoch = base_epoch+new_epoch
            # Log the LR for step to be trained
            writer.add_scalar('lr', sched.get_last_lr()[0], step)
            # Clean the gradient on the parameters
            optim.zero_grad()
            # Get and add noise to x_0, and get the ground truths
            noise = net_model.get_noise(x_0.shape[0], device)
            x_T = net_model.add_noise(x_0, noise)
            x_gt_all = net_model.get_multi_ground_truth(x_0, noise)
            # Predict images
            x_pred_all = net_model(x_T, x_gt_all)
            # Calculate and log the MES loss
            loss = F.mse_loss(x_pred_all, x_gt_all, reduction='none').mean(dim=(0, 2, 3, 4))
            for idx in range(len(loss)):
                writer.add_scalar(f'loss-{idx}', loss[idx], step)
            loss = (loss_coef*loss).sum()
            writer.add_scalar('loss-avg', loss, step)
            # Backprop and optimize, and update the LR
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            # Do EMA if necessary
            if ema_model is not None: ema(net_model, ema_model, FLAGS.ema_decay)
            # Update the status bar
            pbar_postfix['epoch'] = epoch
            pbar_postfix['loss'] = '%.2e' % loss
            pbar_postfix['lr'] = '%.2e' % sched.get_last_lr()[0]
            pbar.set_postfix(pbar_postfix, refresh=True)

            # sample
            if FLAGS.sample_step > 0 and (step+1) % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    pred_x_0 = net_model(sample_noise)[:, 0, ...].clip(-1, 1)
                    grid = (make_grid(pred_x_0) + 1) / 2
                    path = os.path.join(FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()
                if ema_model is not None:
                    ema_model.eval()
                    with torch.no_grad():
                        ema_pred_x_0 = ema_model(sample_noise)[:, 0, ...].clip(-1, 1)
                        ema_grid = (make_grid(ema_pred_x_0) + 1) / 2
                        ema_path = os.path.join(FLAGS.logdir, 'sample', '%d_ema.png' % step)
                        save_image(ema_grid, ema_path)
                        writer.add_image('sample_EMA', ema_grid, step)
                    ema_model.train()

            # save
            if FLAGS.save_step > 0 and (step+1) % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict() \
                                 if ema_model is not None else None,
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step, 'epoch': epoch,
                    'sample_noise': sample_noise,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt.pt'))

            # evaluate
            if FLAGS.eval_step > 0 and (step+1) % FLAGS.eval_step == 0:
                net_IS, net_FID, _ = _eval(net_model)
                if ema_model is not None:
                    ema_IS, ema_FID, _ = _eval(ema_model)
                metrics = {'IS': net_IS[0], 'IS_std': net_IS[1], 'FID': net_FID}
                if ema_model is not None:
                    metrics.update({'IS_EMA': ema_IS[0], 'IS_std_EMA': ema_IS[1], 'FID_EMA': ema_FID})
                pbar.write("%d/%d " % (step, FLAGS.total_steps)+", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()


def evaluate():
    # model setup
    with open(FLAGS.net_cfg) as f:
        model_cfg = json.loads(f.read())
    model = DenoisingNet(T=FLAGS.T, tau=FLAGS.tau, **model_cfg,
                         img_size=FLAGS.img_size, img_ch=FLAGS.img_ch,
                         beta_1=FLAGS.beta_1, beta_T=FLAGS.beta_T).to(device)
    if FLAGS.parallel:
        model = torch.nn.DataParallel(model)

    # load model and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    model.load_checkpoint(ckpt['net_model'])
    (IS, IS_std), FID, samples = _eval(model)
    print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(torch.tensor(samples[:256]), os.path.join(FLAGS.logdir, 'samples.png'), nrow=16)

    if ckpt['ema_model'] is not None:
        model.load_checkpoint(ckpt['ema_model'])
        (IS, IS_std), FID, samples = _eval(model)
        print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
        save_image(torch.tensor(samples[:256]), os.path.join(FLAGS.logdir, 'samples_ema.png'), nrow=16)


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    if FLAGS.eval:
        evaluate()
    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)