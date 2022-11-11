import torchvision
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import Conv2d,MaxPool2d,Sequential
from torch import nn
from model import *
from torch.utils.tensorboard import SummaryWriter
train_data=torchvision.datasets.CIFAR10(root="../data",train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_data=torchvision.datasets.CIFAR10(root="../data",train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
train_data_size=len(train_data)
test_data_size=len(test_data)#获取数据集长度
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))
#利用Dataloader加载数据集
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)
pcz=PCZ()#实例化网络
loss_fn=nn.CrossEntropyLoss()
learning_rate=0.01
optimizer=torch.optim.SGD(pcz.parameters(),lr=learning_rate)
total_train_step=0
total_test_step=0
epoch=10
writer= SummaryWriter("../logs_train")
for i in range(epoch):
    print("---------第{}轮训练开始--------".format(i+1))
    pcz.train()
    for data in train_dataloader:
        imgs,targets=data
        output=pcz(imgs)
        loss=loss_fn(output,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step=total_train_step+1
        if total_train_step%100==0:
            print("训练次数:{},Loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)
    #测试步骤开始
    pcz.eval()
    total_test_loss=0
    total_acc=0
    with torch.no_grad():
        for test in test_dataloader:
            imgs,targets=test
            output=pcz(imgs)
            loss=loss_fn(output,targets)
            total_test_loss=total_test_loss+loss
            acc=(output.argmax(1)==targets).sum()
            total_acc=total_acc+acc
        print("整体测试集上的Loss:{}".format(total_test_loss))
        print("整体测试集上的正确率:{}".format(total_acc/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_acc",total_acc/test_data_size,total_test_step)
    total_test_loss=total_test_loss+1
    #保存模型
    torch.save(pcz,"pcz_{}.pth".format(i))
    print("模型已保存")
writer.close()
#scalar标量 vector向量 tensor张量



