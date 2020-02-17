#coding:utf-8
#This file is used for validating the trained model

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from load_mnist import load_mnist
from train import Net

def validation():
    model=torch.load('./model/mnist_model.pkl')
    dataloader=load_mnist()
    loss_value=np.zeros((dataloader.train_num,),dtype=np.float32)
    cnt=0
    for data,label in dataloader.val_loader():
        data=data/255.0 #data normalization
        data=data-0.5
        data=torch.from_numpy(data)
        label=torch.LongTensor(label)
        output=model(data)
        loss=F.nll_loss(output,label)
        loss_value[cnt]=loss
        cnt+=1
    print 'validation loss:',np.mean(loss_value)

if __name__=='__main__':
    validation()
