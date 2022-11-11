import torch
from PIL import Image
from torch import nn
import torchvision

img_path="../data/dog.png"
image=Image.open(img_path)
# print(image)
image=image.convert('RGB')
transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                          torchvision.transforms.ToTensor()])
image=transform(image)
print(image.shape)
# class PCZ(nn.Module):
#     def __init__(self):
#         super(PCZ, self).__init__()
#         self.model=nn.Sequential(
#             nn.Conv2d(3,32,5,1,2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32,32,5,1,2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32,64,5,1,2),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(64*4*4,64),
#             nn.Linear(64,10)
#         )
#     def forward(self,x):
#         x=self.model(x)
#         return x
#加载模型
model=torch.load("pcz_9.pth")
print(model)
image=torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    output=model(image)
print(output)
print(output.argmax(1))