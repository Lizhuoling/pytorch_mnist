#coding:utf-8
#This file is used for training model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import time
import os
from load_mnist import load_mnist

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #self.conv=nn.Conv2d(in_channels=1,out_channels=10,kernel_size=3)
        #self.mp=nn.MaxPool2d(kernel_size=2)
        self.fc1=nn.Linear(in_features=28*28,out_features=10*10)
        self.fc2=nn.Linear(in_features=10*10,out_features=10)

    def forward(self,x):
        batch_size=x.shape[0]
        #x=F.relu(self.mp(self.conv(x)))
        x=x.view(batch_size,-1)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        x=F.log_softmax(x,dim=1)
        return x

def train(epoches):
    model=Net()
    op=optim.SGD(model.parameters(),lr=1e-2,momentum=0.5)
    dataloader=load_mnist()
    loss_value=np.zeros((dataloader.train_num,),dtype=np.float32)
    print 'start training'
    start_time=time.time()
    for epoch in range(epoches):
        cnt=0
        for data,label in dataloader.train_loader():
            data=data/255.0 #data normalization
            data=data-0.5
            data=torch.from_numpy(data)
            label=torch.LongTensor(label)
            output=model(data)
            loss=F.nll_loss(output,label)
            loss_value[cnt]=loss
            cnt+=1
            op.zero_grad()
            loss.backward()
            op.step()
        now_time=time.time()
        print 'epoch',epoch+1,'time',now_time-start_time,'loss',np.mean(loss_value)
    if os.path.exists('./model/') is False:
        os.mkdir('./model/')
    torch.save(model,'./model/mnist_model.pkl')

if __name__=='__main__':
    train(epoches=15)
