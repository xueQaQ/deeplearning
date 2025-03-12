import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()

        #卷积 + 池化操作 直接调用函数
        self.c1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        self.sig = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.c3 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2,stride=2)

        self.flatten = nn.Flatten()

        #三层全连接层
        self.f5 = nn.Linear(400,120)
        self.f6 = nn.Linear(120,84)
        self.f7 = nn.Linear(84,10)
    
    #前向传播
    def forward(self, x):
        x1 = self.sig(self.c1(x))
        x2 = self.s2(x1)
        x3 = self.sig(self.c3(x2))
        x4 = self.s4(x3)
        x5 = self.flatten(x4)
        x6 = self.f5(x5)
        x7 = self.f6(x6)
        x8 = self.f7(x7)
        return x8 


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = LeNet().to(device)
    print(summary(model,(1,28,28)))




