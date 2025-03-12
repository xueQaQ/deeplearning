from time import time
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch

import torch.utils.data as Data
import numpy as np

import matplotlib.pyplot as plt

from model import LeNet

import torch.nn as nn
import time


def train_val_data_process():
    train_data = FashionMNIST(root="./data", 
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224),    # 将28x28图像上采样到224x224
                                                        transforms.ToTensor()           # 转换为张量并归一化到[0,1]
                                                        ]),
                          download=True)
    
     # 划分训练集和验证集（8:2比例）
    train_data, val_data = Data.random_split(train_data,
                                             [round(0.8*len(train_data)),round(0.2*len(train_data))])
    
    # 创建数据加载器
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=128,
                                       shuffle=True,       #shuffle 是否打乱顺序， True，打乱，False，不打乱
                                       num_workers=8)      #使用8个子进程加载数据

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=8)
    return train_dataloader,val_dataloader


def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    #检测是否有可用的 GPU，如果有则使用 CUDA，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #优化器设置，使用 Adam 优化器，并指定学习率为 0.001
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    #损失函数定义，交叉熵损失
    criterion = nn.CrossEntropyLoss()
    #将模型加载至设备中
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    
    # 记录训练过程指标
    train_loss_all = []
    val_loss_all = []

    train_acc_all = []
    val_acc_all = []

    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch,num_epochs=1))
        print("_"*10)

        train_loss = 0.0
        train_corrects = 0.0 

        val_loss = 0.0
        val_corrects = 0.0 

        train_num = 0
        val_num = 0

        for step, (b_x,b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.train()

            ouput = model(b_x)

            pre_lab = torch.argmax(ouput,dim=1)

            #计算每一个batch的损失函数
            loss = criterion(ouput,b_y)

            #梯度置为0
            optimizer.zero_grad()

            loss.backward()
            #根据梯度信息更新网络参数，降低loss函数计算值的作用
            optimizer.step()

            train_loss+= loss.item() * b_x.size(0)

            train_corrects += torch.sum(pre_lab == b_y.data)

            train_num += b_x.size(0)

        for step, (b_x,b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.val()

            ouput = model(b_x)

            pre_lab = torch.argmax(ouput,dim=1)

            #计算每一个batch的损失函数
            loss = criterion(ouput,b_y)

            #梯度置为0
            optimizer.zero_grad()
            val_loss+= loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)


        #计算epoch指标
        epoch_loss = train_loss/train_num
        epoch_acc = train_corrects.double()/train_num






# train_data,val_dataloader = train_val_data_process()