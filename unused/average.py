from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image
import torch

dataset_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.Resize(32),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = CIFAR10(root='./data', train=True, download=True, transform=dataset_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=True)

avg_img, batch_cnt = torch.zeros(3, 32, 32), 0
for x, _ in iter(dataloader):
    assert x.shape[0] == 32
    x = x.mean(dim=0)
    avg_img += x
    batch_cnt += 1
avg_img = avg_img / batch_cnt

avg_img = (avg_img.clip(-1, 1)+1.0)/2.0
avg_img = avg_img.mul(255).add_(0.5).clamp_(0, 255)
avg_img = avg_img.permute(1, 2, 0).to(torch.uint8).numpy()
Image.fromarray(avg_img).save('cifar10_average.png')