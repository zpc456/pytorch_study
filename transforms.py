from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
img_path="hymenoptera_data/hymenoptera_data/train/bees/17209602_fe5a5a746f.jpg"
img=Image.open(img_path)
writer=SummaryWriter("logs")

#ToTensor的使用：将图片转换成tensor类型（张量）
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
print(img_tensor)#将图片转换成tensor
writer.add_image("ToTensor",img_tensor)

#Normalize  output[channel] = (input[channel] - mean[channel]) / std[channel]
print(img_tensor[0][0][0])
trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm=trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm)

#Risize
print(img.size)
trans_resize=transforms.Resize((1000,1000))
img_resize=trans_resize(img)
print(img_resize)
img_resize=trans_totensor(img_resize)#PIL->tensor
writer.add_image("Risize",img_resize,0)

writer.close()
