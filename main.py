import copy
import json
import os
# import random
import warnings
from absl import app, flags

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange

from model import DenoisingNet
from score.both import get_inception_and_fid_score


FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')

# Denoising Diffusion Network
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 100, help='total diffusion steps')
flags.DEFINE_string('net_cfg', './config/net.json', help='path of network conf file (JSON)')

# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate (non-positive means disable ema)")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
flags.DEFINE_bool('end2end', True, help='enable end-to-end training')
flags.DEFINE_bool('cont_from_ckpt', False, help='continue training from the checkpoint')

# Logging & Sampling
flags.DEFINE_string('logdir', './ckpts', help='log directory')
flags.DEFINE_bool('log_timestep', True, help='log the loss each timestep')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')

# Evaluation
flags.DEFINE_integer('save_step', 50, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')


# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    epoch_cnt = 0
    while True:
        for x, y in iter(dataloader):
            yield x, epoch_cnt
        epoch_cnt += 1


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def _eval(model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            noises = model.get_noise(batch_size, device, 'all')
            batch_images = model(noises.to(device)).clip(-1, 1).cpu()
            # Data range from [-1, 1] to [0, 1]
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def train():
    # dataset
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # Pixel range [0, 1.0]
            transforms.ToTensor(),
            # Pixel range [-0.5, 0.5]->[-1.0, 1.0]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    with open(FLAGS.net_cfg) as f:
        model_cfg = json.loads(f.read())

    # model setup
    net_model = DenoisingNet(T=FLAGS.T, cfg=model_cfg['cfg'], cfg_ratio=model_cfg['cfg_ratio'],
                             input_size=FLAGS.img_size, input_ch=3,
                             beta_1=FLAGS.beta_1, beta_T=FLAGS.beta_T).to(device)
    ema_model = copy.deepcopy(net_model) if FLAGS.ema_decay > 0 else None
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        if ema_model is not None:
            ema_model = torch.nn.DataParallel(ema_model)
        net_model = torch.nn.DataParallel(net_model)

    # sample_noises for sampling using net_model during training process
    sample_noises = net_model.get_noise(FLAGS.sample_size, device, 'all')
    # show model size
    model_size = 0
    for param in net_model.parameters():
        if not param.requires_grad: continue
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    start_step, base_epoch = -1, 0
    if FLAGS.cont_from_ckpt:
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
        net_model.load_state_dict(ckpt['net_model'])
        if ema_model is not None:
            if ckpt['ema_model'] is None: ema_model = None
            else: ema_model.load_state_dict(ckpt['ema_model'])
        sched.load_state_dict(ckpt['sched'])
        optim.load_state_dict(ckpt['optim'])
        start_step = ckpt['step']
        base_epoch = ckpt['epoch']
        sample_noises = ckpt['sample_noises']

    # log setup
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'), exist_ok=True)
    purge_step = start_step if FLAGS.cont_from_ckpt else None
    writer = SummaryWriter(FLAGS.logdir, purge_step=purge_step)
    if not FLAGS.cont_from_ckpt:
        grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
        writer.add_image('real_sample', grid)
        writer.flush()
    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())

    pbar_postfix = {'epoch': 0, 'loss': '0', 'lr': '0'}
    if not FLAGS.end2end:
        pbar_postfix.update({'layer': '0/0', 'layer_loss': '0'})

    # start training
    with trange(start_step+1, FLAGS.total_steps, dynamic_ncols=True) as pbar:
        # Do optimization on the full net with a mini-batch each step
        for step in pbar:
            # train
            x_0, new_epoch = next(datalooper)
            epoch = base_epoch+new_epoch
            pbar_postfix['epoch'] = epoch
            x_0 = x_0.to(device)
            timesteps, losses = list(range(2, FLAGS.T+1)) if not FLAGS.end2end else [None], []
            for t in timesteps: # Every layer should be optimized with the same mini-batch
                optim.zero_grad()
                noise = net_model.get_noise(x_0.shape[0], device, mode='single')
                if FLAGS.end2end:
                    x_T = net_model.add_noise(x_0, noise, FLAGS.T)
                    noises = net_model.get_noise(x_0.shape[0], device, 'all', x_T)
                    x_0_pred = net_model(noises)
                    loss = F.mse_loss(x_0_pred, x_0, reduction='none').mean()
                else:
                    pbar_postfix['layer'] = f'{t}/{FLAGS.T}'
                    x_t = net_model.add_noise(x_0, noise, t)
                    noise_pred = net_model(x_t, t)
                    loss = F.mse_loss(noise_pred, noise, reduction='none').mean()
                    pbar_postfix['layer_loss'] = '%.2e' % loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), FLAGS.grad_clip)
                optim.step()
                if not FLAGS.end2end and FLAGS.log_timestep:
                    writer.add_scalar(f'loss-layer_{t}', loss, step)
                losses.append(loss)
                pbar.set_postfix(pbar_postfix, refresh=True)
            avg_loss = sum(losses) / len(losses)
            writer.add_scalar('loss', avg_loss, step)
            pbar_postfix['loss'] = '%.2e' % avg_loss

            writer.add_scalar('lr', sched.get_last_lr()[0], step)
            pbar_postfix['lr'] = '%.2e' % sched.get_last_lr()[0]
            sched.step() # Modify the learning rate per step

            if ema_model is not None:
                ema(net_model, ema_model, FLAGS.ema_decay)

            # sample
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = net_model(sample_noises)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict() \
                                 if ema_model is not None else None,
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step, 'epoch': epoch,
                    'sample_noises': sample_noises,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt.pt'))

            # evaluate
            if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                net_IS, net_FID, _ = _eval(net_model)
                if ema_model is not None:
                    ema_IS, ema_FID, _ = _eval(ema_model)
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID
                }
                if ema_model is not None:
                    metrics.update({
                        'IS_EMA': ema_IS[0],
                        'IS_std_EMA': ema_IS[1],
                        'FID_EMA': ema_FID
                    })
                pbar.write(
                    "%d/%d " % (step, FLAGS.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()


def evaluate():
    with open(FLAGS.net_cfg) as f:
        model_cfg = json.loads(f.read())

    # model setup
    model = DenoisingNet(T=FLAGS.T, cfg=model_cfg['cfg'], cfg_ratio=model_cfg['cfg_ratio'],
                         input_size=FLAGS.img_size, input_ch=3,
                         beta_1=FLAGS.beta_1, beta_T=FLAGS.beta_T).to(device)
    if FLAGS.parallel:
        model = torch.nn.DataParallel(model)

    # load model and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    model.load_state_dict(ckpt['net_model'])
    (IS, IS_std), FID, samples = _eval(model)
    print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples.png'),
        nrow=16)

    if ckpt['ema_model'] is not None:
        model.load_state_dict(ckpt['ema_model'])
        (IS, IS_std), FID, samples = _eval(model)
        print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
        save_image(
            torch.tensor(samples[:256]),
            os.path.join(FLAGS.logdir, 'samples_ema.png'),
            nrow=16)


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
