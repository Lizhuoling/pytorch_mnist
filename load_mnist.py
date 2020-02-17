#coding:utf-8
#本文件用于为训练和测试加载数据

import numpy as np
import cv2

class load_mnist(object):
    def __init__(self,data_path='./data/',
            train_data_file='train_data/',
            train_label_file='train_label/',
            val_data_file='val_data/',
            val_label_file='val_label/',
            train_num=45176, #训练数据数量
            val_num=8952    #校验数据数量
            ):
        self.data_path=data_path
        self.train_data_file=train_data_file
        self.train_label_file=train_label_file
        self.val_data_file=val_data_file
        self.val_label_file=val_label_file
        self.train_num=train_num
        self.val_num=val_num
    #训练数据加载器,pytorch卷积输入格式:[B,C,H,W]
    def train_loader(self):
        train_label=np.load(self.data_path+self.train_label_file+'train_label.npy')
        for i in range(self.train_num):
            train_data=cv2.imread(self.data_path+self.train_data_file+'train'+str(i+1)+'.png')
            train_data=train_data[:,:,0]
            train_data=train_data.astype(np.float32)
            data=train_data.reshape(1,1,train_data.shape[0],train_data.shape[1])
            label=[train_label[i]]
            yield data,label
    #校验数据加载器
    def val_loader(self):
        val_label=np.load(self.data_path+self.val_label_file+'val_label.npy')
        for i in range(self.val_num):
            val_data=cv2.imread(self.data_path+self.val_data_file+'val'+str(i+1)+'.png')
            val_data=val_data[:,:,0]
            val_data=val_data.astype(np.float32)
            data=val_data.reshape(1,1,val_data.shape[0],val_data.shape[1])
            label=[val_label[i]]
            yield data,label

if __name__=='__main__':
    a=load_mnist()
    for data,label in a.train_loader():
        print 'label:',label
        print 'data shape',data.shape
        break
    for data,label in a.val_loader():
        print 'label',label
        print 'data shape',label.shape
        break
