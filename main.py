import os, copy
from absl import app
from tqdm import trange

from utility.misc import set_warning_level
set_warning_level()

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusion.model import DenoisingNet

from utility.argument import register_cmd_argument, parse_cmd_argument
from utility.logger import TensorboardLogger
from utility.distributed import get_device_name_and_id_list, init_parallel_env, destroy_parallel_env
from utility.misc import save_img_tensor, load_from_json
from utility.dataset import get_dataloader, get_datalooper
from utility.train import get_optim_and_sched, get_loss_coef, train_one_step
from utility.train import dump_training_data_to_dict, resume_training_data
from utility.sample import sample_once_from_noise
from utility.evaluate import evaluate_model


def construct_end2end_model(kwargs, device):
    model_cfg = load_from_json(kwargs['net_cfg'])
    model = DenoisingNet(device, T=kwargs['T'], tau=kwargs['tau'], **model_cfg,
                         img_size=kwargs['img_size'], img_ch=kwargs['img_ch'],
                         beta_1=kwargs['beta_1'], beta_T=kwargs['beta_T']).to(device)
    return model


def train(rank, kwargs):
    init_parallel_env(rank, kwargs['nprocs'])
    device, device_ids = get_device_name_and_id_list(rank)
    net_model = construct_end2end_model(kwargs, device)
    net_model = DDP(net_model, device_ids=device_ids)
    optim, sched = get_optim_and_sched(net_model, kwargs['lr'], kwargs['warmup'])
    loss_coef = get_loss_coef(net_model, kwargs['layer_loss_factor'])
    # If ema_model is wrapped with DDP, all devices should hold the EMA model 
    # Otherwise, there will be problem of reference counting
    ema_model = copy.deepcopy(net_model) if kwargs['ema_decay'] > 0 else None
    if rank == 0: # Pre-checkpoint-loading preparations for main device only
        # Only main device do sampling, but all devices do distributed evaluation
        sample_noise = net_model.module.get_noise(kwargs['sample_size'])
        # Display the number of parameters
        net_model.module.print_model_size()
    if len(kwargs['resume_from_ckpt']) != 0:
        layer_excluded = list(map(int, kwargs['resume_layer_excluded']))
        ckpt_dict = torch.load(kwargs['resume_from_ckpt'], map_location=device)
        sample_noise, base_step, base_epoch = resume_training_data(net_model, ema_model, optim, sched,
                                                                   sample_noise if rank == 0 else None,
                                                                   ckpt_dict, layer_excluded, 
                                                                   kwargs['resume_without_progress'])
        if rank != 0: del sample_noise
    else: base_epoch, base_step = 0, 0
    if rank == 0: # Post-checkpoint-loading preparations for main device only
        # Only main process logs out training information
        purge_step = base_step-1 if len(kwargs['resume_from_ckpt']) != 0 else None
        logger = TensorboardLogger(kwargs['logdir'], purge_step)
        logger.write_config_bak_file(kwargs)
    assert kwargs['batch_size'] % kwargs['nprocs'] == 0
    dataloader, sampler = get_dataloader(kwargs['img_ch'], kwargs['img_size'],
                                         kwargs['batch_size']//kwargs['nprocs'], kwargs['num_workers'],
                                         kwargs['nprocs'], rank, kwargs['dataset'])
    datalooper = get_datalooper(dataloader, sampler, base_epoch)
    # Initiate progress bar for all devices
    pbar_postfix = {'epoch': 'N/A', 'loss': 'N/A', 'lr': 'N/A'}
    pbar_iterator = trange(base_step, kwargs['total_steps'], dynamic_ncols=True, leave=None, position=rank)
    for step in pbar_iterator:
        # 1 - Train (distributed)
        if rank == 0: logger.set_step(step)
        loss_per_layer, epoch, lr = train_one_step(rank, kwargs['nprocs'], net_model, ema_model,
                                                   datalooper, sched, optim, loss_coef,
                                                   kwargs['grad_clip'], kwargs['ema_decay']) # SYNC
        if rank == 0: # i.e. loss_per_layer is not None
            logger.write_data('lr', lr)
            for layer_idx in range(len(loss_per_layer)):
                logger.write_data('loss', loss_per_layer[layer_idx], str(layer_idx))
            loss_total = (loss_per_layer*loss_coef).sum()
            logger.write_data('loss', loss_total, 'weighted_total')
        pbar_postfix.update({'epoch': epoch, 'lr': '%.2e' % lr, \
                             'loss': ('%.2e' % loss_total) if rank == 0 else 'N/A'})
        pbar_iterator.set_postfix(pbar_postfix, refresh=True) # Update the status bar
        # 2 - Evaluate (distributed)
        if kwargs['eval_step'] > 0 and (step+1) % kwargs['eval_step'] == 0:
            for model, postfix in [(net_model, ''), (ema_model, '_EMA')]:
                if model is None: continue
                (IS, IS_std), FID, _ = evaluate_model(rank, kwargs['nprocs'], model,
                                                      kwargs['num_images'], kwargs['batch_size'],
                                                      kwargs['fid_cache'], kwargs['eval_use_torch'], pbar_bias=1)
                if rank == 0: metrics = {f'IS{postfix}': IS, f'IS_std{postfix}': IS_std, f'FID{postfix}': FID}
            if rank == 0: # Only main device do logging
                pbar_iterator.write("%d/%d " % (step, kwargs['total_steps']) +
                                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                logger.write_data_dict(metrics)
                logger.write_eval_file(metrics)
        # The following action is only done on the main device
        # Local model is used instead of the one wrapped with DDP,
        # to prevent reference counting problem, and hanging on SYNC
        if rank != 0: continue # If not the main device, just skip
        # 3 - Get some samples for visualization
        if kwargs['sample_step'] > 0 and (step+1) % kwargs['sample_step'] == 0:
            for model, postfix in [(net_model, ''), (ema_model, 'EMA')]:
                if model is None: continue
                pred_x_0 = sample_once_from_noise(model.module, sample_noise)
                logger.write_img('sample', pred_x_0, postfix, save_png=True)
        # 4 - Save checkpoint
        if kwargs['save_step'] > 0 and (step+1) % kwargs['save_step'] == 0:
            ckpt_dict = dump_training_data_to_dict(net_model, ema_model, optim, sched, \
                                                   step, epoch, sample_noise)
            torch.save(ckpt_dict, os.path.join(kwargs['logdir'], 'ckpt.pt'))
    if rank == 0: del logger
    destroy_parallel_env()


# rank = ID of CPU/GPU, starts from 0
# Parallel on batch size for training, batch size for one device is batch_size//nprocs
# Parallel on number of images for evaluation, batch size for one device is batch_size
def evaluate(rank, kwargs):
    init_parallel_env(rank, kwargs['nprocs'])
    device, device_ids = get_device_name_and_id_list(rank)
    model = construct_end2end_model(kwargs, device)
    model = DDP(model, device_ids=device_ids)
    ckpt = torch.load(os.path.join(kwargs['logdir'], 'ckpt.pt'), map_location=device)
    for model_name, postfix in [('net_model', ''), ('ema_model', 'EMA')]:
        if ckpt[model_name] is None: continue
        model.module.load_checkpoint(ckpt[model_name])
        (IS, IS_std), FID, sample = evaluate_model(rank, kwargs['nprocs'], model,
                                                   kwargs['num_images'], kwargs['batch_size'],
                                                   kwargs['fid_cache'], kwargs['eval_use_torch'])
        if rank != 0: continue
        print((f'[Model {postfix}]' if len(postfix) != 0 else '[Model]')+
                ' IS: %6.3f (%.3f), FID: %7.3f' % (IS, IS_std, FID))
        sample_filename = f'samples' + (f'_{postfix}.png' if len(postfix) != 0 else '.png')
        sample_path = os.path.join(kwargs['logdir'], sample_filename)
        save_img_tensor(sample, sample_path)
    destroy_parallel_env()


# All options handled by absl are removed from sys_argv
def main(sys_argv):
    # Execute parsing for options handled by absl into dict
    kwargs = parse_cmd_argument()
    if kwargs['train']:
        mp.spawn(train, nprocs=kwargs['nprocs'], args=(kwargs,))
    if kwargs['eval']:
        mp.spawn(evaluate, nprocs=kwargs['nprocs'], args=(kwargs,))
    if not kwargs['train'] and not kwargs['eval']:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    register_cmd_argument('./utility/flag.json')
    app.run(main, argv=None)