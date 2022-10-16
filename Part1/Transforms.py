from torchvision import transforms
from PIL import Image
# transforms.ToTensor->tensor数据类型
# 将PILimage和nump.array转化成tensor，tensor类型的图片在计算机中用一些数据表示

img_path = './Data/hymenoptera_data/train/ants/5650366_e22b7e1065.jpg'
img = Image.open(img_path)
print(img)

trans = transforms.ToTensor()
trans_img = trans(img)
print(trans_img)

