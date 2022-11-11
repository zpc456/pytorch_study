#Define Dataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
training_data=datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data=datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

plt.imshow(training_data[5][0][0,...],cmap='gray')
#plt.show()
print(len(training_data))

print(len(test_data))

#Define Dataloader
from torch.utils.data import DataLoader
train_dataloader=DataLoader(training_data,batch_size=10,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=10,shuffle=True)
a=iter(train_dataloader).__next__()
print(len(a[0]))

#Define Transform
from torchvision.transforms import ToTensor,RandomCrop,Compose

training_data=datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=Compose([RandomCrop(20),ToTensor()])
)

iter(training_data)
plt.imshow(training_data[0][0][0,...])
plt.show()
