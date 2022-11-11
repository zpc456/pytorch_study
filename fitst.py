import torch
x=torch.rand(5,3)
print(x)
y=torch.rand(4,5)
print(y)
print(torch.__version__)
print(torch.cuda.is_available())