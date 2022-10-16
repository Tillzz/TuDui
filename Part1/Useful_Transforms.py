from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("log")
img_path = '../Data/pic.jpeg'
img = Image.open(img_path)

trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
writer.add_image("ToTensor", img_tensor)

print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

writer.close()
