import torchvision

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
# 下载pytorch官方数据集
train_set = torchvision.datasets.CIFAR10(root="../Data/dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="../Data/dataset", train=False, transform=dataset_transform, download=True)

# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

print(test_set[0])
