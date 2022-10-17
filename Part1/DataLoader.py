import torchvision
from torch.utils.data import DataLoader

# 准备的测试数据集

test_data = torchvision.datasets.CIFAR10("../Data/dataset", train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中的第一张图片以及target
img ,target= test_data[0]
print(img.shape)
print(target)

# 这里运行的时候会报错，因为python版本问题，将python版本降到3.8就可以运行成功

for data in test_loader:
     imgs, targets = data
     print(imgs.shape)
     print(targets)