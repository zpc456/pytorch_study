import torch
from torch import nn
class PCZ(nn.Module):
    def __init__(self):
        super(PCZ, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x=self.model(x)
        return x
if __name__ == '__main__':
    pcz=PCZ()
    input=torch.ones((64,3,32,32))
    ouput=pcz(input)
    print(ouput.shape)