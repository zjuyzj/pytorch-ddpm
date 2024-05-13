import torch
import torch.distributed as dist

from score.both import get_inception_and_fid_score
from utility.sample import sample_batch_all


def evaluate_model(rank, nprocs, model, num_img, batch_size, fid_cache, use_torch, pbar_bias=0):
    assert num_img % nprocs == 0
    num_img_local = num_img // nprocs
    img_local = sample_batch_all(model, num_img_local, batch_size, rank, pbar_bias)
    if rank == 0: # Main process (device) for calculating FID and IS
        img_gathered, img_received = [img_local], torch.zeros_like(img_local)
        for i in range(1, nprocs):
            dist.recv(img_received, i)
            img_gathered.append(img_received.clone())
        img_gathered = torch.cat(img_gathered, dim=0).numpy()
        (IS, IS_std), FID = get_inception_and_fid_score(img_gathered, fid_cache, img_gathered.shape[0],
                                                        use_torch=use_torch, verbose=True)
        return (IS, IS_std), FID, torch.tensor(img_gathered[:256])
    else: # Blocking point-to-point sending
        dist.send(img_local, 0)
        return (None, None), None, None