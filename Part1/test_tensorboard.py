from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("../logs")
img_path = './Data/hymenoptera_data/train/ants/0013035.jpg'
img = Image.open(img_path) # 打开图片
img_array= np.array(img) # 将图片转化为np.array
print(type(img_array))
print(img_array.shape)

writer.add_image("test", img_array, 1, dataformats='HWC')

for i in range(100):
    writer.add_scalar('y=2x', 2*i, i) # 将函数图像化

writer.close()