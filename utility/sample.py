import torch
from tqdm import trange


def sample_once_from_noise(model, noise):
    model.eval()
    with torch.no_grad():
        pred_x_0 = model(noise)[:, 0, ...].clip(-1, 1)
    model.train()
    pred_x_0 = (pred_x_0+1)/2
    return pred_x_0


def sample_batch_all(model, num_img, batch_size, rank=0, pbar_bias=0):
    img_all, prompt_str = list(), f"Generating Images (Device {rank})"
    for i in trange(0, num_img, batch_size, desc=prompt_str, leave=False, position=rank+pbar_bias):
        curr_bs = min(batch_size, num_img-i)
        noise = model.module.get_noise(curr_bs)
        img = sample_once_from_noise(model, noise)
        img_all.append(img)
    return torch.cat(img_all, dim=0)