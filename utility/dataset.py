import torch
from torchvision.datasets import CIFAR10, FashionMNIST
from torchvision import transforms


# world_size = Number of nodes * Number of GPUs on a node
# For each GPU, it's number of workers is num_workers 
def get_dataloader(img_ch, img_size, batch_size, num_workers, world_size, rank, dataset='CIFAR10'):
    assert dataset in ['FashionMNIST', 'CIFAR10']
    dataset_norm_factor = tuple([0.5]*img_ch)
    dataset_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.Resize(img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(dataset_norm_factor, dataset_norm_factor)])
    dataset = eval(dataset)(root='./data', train=True, download=True, transform=dataset_transform)
    # For DistributedSampler, shuffle is True by default, when do sampler.set_epoch(epoch) for each epoch
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                                             sampler=sampler, drop_last=True)
    return dataloader, sampler


def get_datalooper(dataloader, sampler, base_epoch):
    # Since granularity of checkpoint saving is step,
    # some step in an unfinished epoch may be lost
    epoch_cnt = base_epoch # Epoch in previous training
    while True: # Infinite loop for next()
        sampler.set_epoch(epoch_cnt)
        for x, _ in iter(dataloader):
            yield x, epoch_cnt
        epoch_cnt += 1
    # It never reaches here!