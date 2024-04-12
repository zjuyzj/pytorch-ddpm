import copy
import json
import os
from random import shuffle
import warnings
from absl import app, flags

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange

from model import DenoisingNet
from loss import VGGLoss
from score.both import get_inception_and_fid_score

# from torchviz import make_dot

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')

# Denoising Diffusion Network
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_string('net_cfg', './config/net/unet_t.json', help='path of network conf file (JSON)')

# Dataset
flags.DEFINE_string('dataset', 'CIFAR10', help='dataset selection (CIFAR10/MNIST/FashionMNIST)')

# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800, help='total training steps (in a step, all timesteps or layers is optimized once)')
flags.DEFINE_integer('img_size', 32, help='size of image fed into the network (in 2^k for UNet) as well as the output')
flags.DEFINE_integer('img_ch', 3, help='image channel (must match the train set)')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup (Unit is timestep if not set sched_lr_on_timestep, else step)')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate (non-positive means disable ema)")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
flags.DEFINE_bool('end2end', False, help='enable end-to-end training')
flags.DEFINE_spaceseplist('layer_trained', '', help='specify the layer(s) to be trained')
flags.DEFINE_bool('resume_from_ckpt', False, help='continue training from the checkpoint')
flags.DEFINE_bool('resume_without_progress', False, help='load checkpoint but ignore the saved progress')
flags.DEFINE_spaceseplist('resume_layer_excluded', '', help='layer not loaded when resume_from_ckpt is True')
# Note: all samples in a mini-batch share the same timestep in this implementation although time-embedding
# is enabled, which is different from the reference code, but it causes little performance loss
# 'resample_on_timestep', 'sched_lr_on_timestep', 'randomized_timestep' and 'ema_on_timestep'
# may be set to True when UNet_T is used to get the same performance of DDPM's baseline
flags.DEFINE_bool('resample_on_timestep', True, help='for different layers in one training step, use resampled mini-batch from training set')
flags.DEFINE_bool('sched_lr_on_timestep', True, help='update the learning rate on timestep rather than the whole training step')
flags.DEFINE_bool('randomized_timestep', True, help='shuffle the timestep walked through in one training step')
# If the model's different timesteps(layers) are not shared, 'ema_on_timestep'
# is recommended to set to False which can save time on EMA parameter copying
flags.DEFINE_bool('ema_on_timestep', True, help='do EMA whenever a different timestep is optimized in the whole training step')
flags.DEFINE_float('lambda_vgg', 0.1, help='ratio of VGG perceptual loss for end2end training')
flags.DEFINE_float('lambda_mse', 10, help='ratio of pixel-wise MES loss for end2end training')

# Logging & Sampling
flags.DEFINE_string('logdir', './ckpts', help='log directory')
flags.DEFINE_bool('log_timestep', True, help='log the loss each timestep')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1, help='frequency of sampling')
flags.DEFINE_integer('tau_S', -1, help='DDIM sampling steps (non-positive means disable DDIM sampling)')

# Evaluation
flags.DEFINE_integer('save_step', 5, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache (must match the train set)')


# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Only parameters which require gradient are copied to prevent 
# possible loss of floating-point precision of constants
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
        for x, y in iter(dataloader):
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
            x_T, Z = model.get_noise(batch_size, device, 'all')
            batch_images = model(x_T, Z).clip(-1, 1).cpu()
            # Data range from [-1, 1] to [0, 1]
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def destroy_and_recover(model, x_0, timesteps, device, destroy_ratio):
    # DEBUG: Check sematic consistency for end2end network before it is further trained
    # Do this AFTER initialized with layer-by-layer pretrained weights and BEFORE fine-tuning 
    save_image((make_grid(x_0)+1)/2, 'x_0.png')
    print(f'[INFO] ground truth x_0.png is saved')
    t = timesteps[int((len(timesteps)-1)*destroy_ratio)]
    noise_t = model.get_noise(x_0.shape[0], device, mode='noise_t', t=t)
    x_t = model.add_noise(x_0, noise_t, t)
    save_image((make_grid(x_t)+1)/2, f'x_{t}.png')
    print(f'[INFO] noisy image x_{t}.png is saved')
    Z = model.get_noise(x_0.shape[0], device, mode='Z')
    with torch.no_grad():
        x_0_pred = model(x_t, Z, mode='all', t=t)
    save_image((make_grid(x_0_pred)+1)/2, 'x_0_pred.png')
    print(f'[INFO] recovered x_0_pred.png is saved')
    exit()


def train():
    # dataset setup
    dataset_norm_factor = tuple([0.5]*FLAGS.img_ch)
    dataset_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.Resize(FLAGS.img_size),
                                            # Pixel range [0, 1.0]
                                            transforms.ToTensor(),
                                            # Pixel range [-0.5, 0.5]->[-1.0, 1.0]
                                            transforms.Normalize(dataset_norm_factor, dataset_norm_factor)])
    assert FLAGS.dataset in ['MNIST', 'FashionMNIST', 'CIFAR10']
    dataset = eval(FLAGS.dataset)(root='./data', train=True, download=True, transform=dataset_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    with open(FLAGS.net_cfg) as f:
        model_cfg = json.loads(f.read())

    # model setup
    net_model = DenoisingNet(T=FLAGS.T, **model_cfg, img_size=FLAGS.img_size, img_ch=FLAGS.img_ch,
                             beta_1=FLAGS.beta_1, beta_T=FLAGS.beta_T, tau_S=FLAGS.tau_S).to(device)
    ema_model = copy.deepcopy(net_model) if FLAGS.ema_decay > 0 else None
    if FLAGS.end2end: vgg_criterion = VGGLoss().to(device)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        if ema_model is not None:
            ema_model = torch.nn.DataParallel(ema_model)
        net_model = torch.nn.DataParallel(net_model)

    # Noises (x_T, Z) for sampling during training process
    sample_noises = net_model.get_noise(FLAGS.sample_size, device, 'all')
    # show model size
    model_size = 0
    for param in net_model.parameters():
        if not param.requires_grad: continue
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    start_step, base_epoch = -1, 0
    if FLAGS.resume_from_ckpt:
        layer_excluded = list(map(int, FLAGS.resume_layer_excluded))
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'), map_location=device)
        net_model.load_checkpoint(ckpt['net_model'], layer_excluded)
        if ema_model is not None:
            if ckpt['ema_model'] is None: ema_model = None
            else: ema_model.load_checkpoint(ckpt['ema_model'], layer_excluded)
        if not FLAGS.resume_without_progress:
            sched.load_state_dict(ckpt['sched'])
            optim.load_state_dict(ckpt['optim'])
            start_step = ckpt['step']
            base_epoch = ckpt['epoch']
            sample_noises = ckpt['sample_noises']

    # log setup
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'), exist_ok=True)
    purge_step = start_step if FLAGS.resume_from_ckpt else None
    writer = SummaryWriter(FLAGS.logdir, purge_step=purge_step)
    if not FLAGS.resume_from_ckpt:
        grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
        writer.add_image('real_sample', grid)
        writer.flush()
    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())

    pbar_postfix = {'epoch': 'N/A', 'loss': 'N/A', 'lr': 'N/A'}
    if not FLAGS.end2end:
        pbar_postfix.update({'layer': 'N/A', 'layer_loss': 'N/A'})

    # start training
    with trange(start_step+1, FLAGS.total_steps, dynamic_ncols=True) as pbar:
        timesteps = net_model.get_t_series()
        if len(FLAGS.layer_trained) != 0:
            layer_trained_idx = set()
            for i, layer_id in enumerate(FLAGS.layer_trained):
                assert isinstance(layer_id, str)
                if '-' in layer_id:
                    layer_range = list(map(int, layer_id.split('-')))
                    assert len(layer_range) == 2
                    layer_s, layer_e = layer_range
                    assert layer_s > 0 and layer_e <= len(timesteps)
                    assert layer_s <= layer_e
                    for i in range(layer_s, layer_e+1):
                        layer_trained_idx.add(int(i)-1)
                else: # Convert layer ID to layer index
                    layer_idx = int(layer_id)-1
                    assert layer_idx >= 0 and layer_idx < len(timesteps)
                    layer_trained_idx.add(layer_idx)
            layer_trained_idx = sorted(layer_trained_idx)
            timesteps = [timesteps[idx] for idx in layer_trained_idx]
        # If `layer_trained` is applied, do end2end training from active layer with the maximum ID down to
        # the first layer. Otherwise, timesteps[-1] equals FLAGS.T whether DDIM sampling is enabled or not
        if FLAGS.end2end: timesteps = [timesteps[-1]]
        for step in pbar: # train
            losses = list() # Store the loss for each timesteps
            # Record the initial learning rate per step
            writer.add_scalar('lr', sched.get_last_lr()[0], step)
            if FLAGS.randomized_timestep: shuffle(timesteps)
            for t in timesteps:
                # If not use `resample_on_timestep`, every layer
                # should be optimized with the same mini-batch 
                if FLAGS.resample_on_timestep or t == timesteps[0]:
                    x_0, new_epoch = next(datalooper)
                    x_0 = x_0.to(device)
                    epoch = base_epoch+new_epoch
                    pbar_postfix['epoch'] = epoch
                # destroy_and_recover(net_model, x_0, net_model.get_t_series(), device, 0.2)
                optim.zero_grad()
                noise_t = net_model.get_noise(x_0.shape[0], device, mode='noise_t', t=t)
                x_t = net_model.add_noise(x_0, noise_t, t)
                if FLAGS.end2end:
                    Z = net_model.get_noise(x_0.shape[0], device, mode='Z')
                    x_0_pred = net_model(x_t, Z, mode='all', t=t)
                    vgg_loss = vgg_criterion(x_0_pred, x_0)
                    mse_loss = F.mse_loss(x_0_pred, x_0, reduction='none').mean()
                    loss = FLAGS.lambda_vgg*vgg_loss+FLAGS.lambda_mse*mse_loss
                else:
                    pbar_postfix['layer'] = f'{t}/{FLAGS.T}'
                    noise_pred_t = net_model(x_t, None, mode='single', t=t)
                    # graph_param = dict(list(net_model.named_parameters()))
                    # graph = make_dot(noise_pred_t, params=graph_param)
                    # graph.render('graph', format='png')
                    loss = F.mse_loss(noise_pred_t, noise_t, reduction='none').mean()
                    pbar_postfix['layer_loss'] = '%.2e' % loss
                loss.backward()
                if not FLAGS.end2end and FLAGS.log_timestep:
                    writer.add_scalar(f'loss-timestep_{t}', loss, step)
                losses.append(loss)
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
                optim.step()
                if FLAGS.sched_lr_on_timestep:
                    sched.step() 
                    pbar_postfix['lr'] = '%.2e' % sched.get_last_lr()[0]
                if FLAGS.ema_on_timestep:
                    if ema_model is not None:
                        ema(net_model, ema_model, FLAGS.ema_decay)
                # Refresh the updated status as fast as possible
                pbar.set_postfix(pbar_postfix, refresh=True)
            avg_loss = sum(losses) / len(losses)
            writer.add_scalar('loss', avg_loss, step)
            pbar_postfix['loss'] = '%.2e' % avg_loss

            if not FLAGS.sched_lr_on_timestep:
                sched.step()
                pbar_postfix['lr'] = '%.2e' % sched.get_last_lr()[0]

            # Refresh the status bar for post-step infos
            pbar.set_postfix(pbar_postfix, refresh=True)

            if not FLAGS.ema_on_timestep:
                if ema_model is not None:
                    ema(net_model, ema_model, FLAGS.ema_decay)

            # sample
            if FLAGS.sample_step > 0 and (step+1) % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    pred_x_0 = net_model(*sample_noises).clip(-1, 1)
                    grid = (make_grid(pred_x_0) + 1) / 2
                    path = os.path.join(FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()
                if ema_model is not None:
                    ema_model.eval()
                    with torch.no_grad():
                        ema_pred_x_0 = ema_model(*sample_noises).clip(-1, 1)
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
                    'sample_noises': sample_noises,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt.pt'))

            # evaluate
            if FLAGS.eval_step > 0 and (step+1) % FLAGS.eval_step == 0:
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
    model = DenoisingNet(T=FLAGS.T, **model_cfg, img_size=FLAGS.img_size, img_ch=FLAGS.img_ch,
                         beta_1=FLAGS.beta_1, beta_T=FLAGS.beta_T, tau_S=FLAGS.tau_S).to(device)
    if FLAGS.parallel:
        model = torch.nn.DataParallel(model)

    # load model and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    model.load_checkpoint(ckpt['net_model'])
    (IS, IS_std), FID, samples = _eval(model)
    print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples.png'),
        nrow=16)

    if ckpt['ema_model'] is not None:
        model.load_checkpoint(ckpt['ema_model'])
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
