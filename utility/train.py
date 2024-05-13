import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist


def do_ema(source_model, target_model, decay: float):
    source_dict = source_model.module.state_dict()
    target_dict = target_model.module.state_dict()
    for key, param in source_model.module.named_parameters():
        if not param.requires_grad: continue
        new_param = target_dict[key].data*decay+source_dict[key].data*(1-decay)
        target_dict[key].data.copy_(new_param)


def get_optim_and_sched(model, lr, warmup):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    lr_lambda = lambda step: min(step, warmup)/warmup
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
    return optim, sched


# Calculate coefficient for intermediate losses
def get_loss_coef(model, layer_loss_factor: float):
    loss_coef = 1.0 / model.module.get_alphas_bar()
    loss_coef = layer_loss_factor*loss_coef.to(model.module.get_device())
    return loss_coef


def train_one_step(rank, nprocs, model, ema_model, datalooper, sched, optim, loss_coef, grad_clip, ema_decay):
    x_0, epoch = next(datalooper)
    x_0 = x_0.to(model.module.get_device())
    curr_lr = sched.get_last_lr()[0]
    optim.zero_grad()
    noise = model.module.get_noise(x_0.shape[0])
    x_T = model.module.add_noise(x_0, noise)
    x_gt_all = model.module.get_multi_ground_truth(x_0, noise)
    x_pred_all = model(x_T, x_gt_all)
    loss = F.mse_loss(x_pred_all, x_gt_all, reduction='none')
    loss = loss.mean(dim=(0, 2, 3, 4)) # Averaged on (B, C, H, W)
    if rank == 0: # Gather loss per layer, blocked
        loss_received = torch.zeros_like(loss)
        loss_per_layer = loss
        for i in range(1, nprocs):
            dist.recv(loss_received, i)
            loss_per_layer += loss_received
        loss_per_layer /= nprocs
    else: dist.send(loss, 0)
    loss = (loss_coef*loss).sum()
    loss.backward() # Implicit synchronization here!
    clip_grad_norm_(model.parameters(), grad_clip)
    optim.step() # Do optimization
    sched.step() # Step the LR
    if ema_model is not None: do_ema(model, ema_model, ema_decay)
    if rank == 0: return loss_per_layer, epoch, curr_lr
    else: return None, epoch, curr_lr


def dump_training_data_to_dict(model, ema_model, optim, sched, step, epoch, sample_noise):
    ckpt_dict = dict()
    ckpt_dict['net_model'] = model.module.state_dict()
    ckpt_dict['ema_model'] = ema_model.module.state_dict() \
                             if ema_model is not None else None
    ckpt_dict['sched'] = sched.state_dict()
    ckpt_dict['optim'] = optim.state_dict()
    ckpt_dict.update({'step': step, 'epoch': epoch})
    ckpt_dict['sample_noise'] = sample_noise
    return ckpt_dict


def resume_training_data(model, ema_model, optim, sched, sample_noise,
                         ckpt_dict, layer_excluded, is_without_progress):
    base_step, base_epoch = 0, 0
    model.module.load_checkpoint(ckpt_dict['net_model'], layer_excluded)
    if ckpt_dict['ema_model'] is not None:
        assert ema_model is not None
        ema_model.module.load_checkpoint(ckpt_dict['ema_model'], layer_excluded)
    if not is_without_progress:
        sched.load_state_dict(ckpt_dict['sched'])
        optim.load_state_dict(ckpt_dict['optim'])
        base_step = ckpt_dict['step'] + 1
        base_epoch = ckpt_dict['epoch']
        sample_noise = ckpt_dict['sample_noise']
    return sample_noise, base_step, base_epoch