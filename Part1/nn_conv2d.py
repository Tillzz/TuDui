import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch import nn

dataset = torchvision.datasets.CIFAR10("../Data/dataset", train=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


class TuDui(nn.Module):
    def __init__(self):
        super(TuDui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


tudui = TuDui()
print(tudui)


for data in dataloader:
    imgs, target = data
    output = tudui(imgs)