from PIL import Image
from torchvision import transforms

img_path = "./Data/1.jpeg"
img = Image.open(img_path)
print(img)
