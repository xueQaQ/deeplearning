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
                          transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                          download=True)

    train_data, val_data = Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=8)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=8)
    return train_data,val_dataloader


def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    train_loss_all = []
    val_loss_all = []

    train_acc_all = []
    val_acc_all = []

    since = time.time()






# train_data,val_dataloader = train_val_data_process()