#coding:utf-8
#This file is for preprocessing the training and validating data.

import torch
import numpy as np
import gzip
import cv2
import os
import sys
import shutil

class prepare_mnist(object):
    def __init__(self,data_path='./data/',
            train_data_name='train-images-idx3-ubyte.gz',
            train_label_name='train-labels-idx1-ubyte.gz',
            val_data_name='t10k-images-idx3-ubyte.gz',
            val_label_name='t10k-labels-idx1-ubyte.gz',
            train_data_file='train_data/',
            train_label_file='train_label/',
            val_data_file='val_data/',
            val_label_file='val_label/',
            train_num=45176, #The number of data for training
            val_num=8952    #The number of data for validation
            ):
        self.data_path=data_path
        self.train_data_name=train_data_name
        self.train_label_name=train_label_name
        self.val_data_name=val_data_name
        self.val_label_name=val_label_name
        self.train_data_file=train_data_file
        self.train_label_file=train_label_file
        self.val_data_file=val_data_file
        self.val_label_file=val_label_file
        self.train_num=train_num
        self.val_num=val_num
    #Transform the binary file into numpy array
    def extract_data(self,target,num_data,head_size,data_size):
        with gzip.open(self.data_path+target) as bytestream:
            bytestream.read(head_size)
            buf=bytestream.read(data_size*num_data)
            data=np.frombuffer(buf,dtype=np.uint8)
        return data
    #Generate the training data
    def generate_train(self):
        #Read the data and labels
        data=self.extract_data(self.train_data_name,self.train_num,16,28*28)
        label=self.extract_data(self.train_label_name,self.train_num,8,1)
        data=data.reshape(self.train_num,28,28)
        label=label.reshape(self.train_num)
        #Check whether the saving target files exist
        if os.path.exists(self.data_path+self.train_data_file) is False:
            os.mkdir(self.data_path+self.train_data_file)
        if os.path.exists(self.data_path+self.train_label_file) is False:
            os.mkdir(self.data_path+self.train_label_file)
        #Save the labels
        np.save(self.data_path+self.train_label_file+'train_label.npy',label)
        #Save the images
        for i in range(data.shape[0]):
            cv2.imwrite(self.data_path+self.train_data_file+'train'+str(i+1)+'.png',data[i,:,:])
        print 'Generating training data has been completed'
    #Generate the validation data
    def generate_val(self):
        #Read the data and labels
        data=self.extract_data(self.val_data_name,self.val_num,16,28*28)
        label=self.extract_data(self.val_label_name,self.val_num,8,1)
        data=data.reshape(self.val_num,28,28)
        label=label.reshape(self.val_num)
        #Check whether the saving target files exist
        if os.path.exists(self.data_path+self.val_data_file) is False:
            os.mkdir(self.data_path+self.val_data_file)
        if os.path.exists(self.data_path+self.val_label_file) is False:
            os.mkdir(self.data_path+self.val_label_file)
        #Save the labels
        np.save(self.data_path+self.val_label_file+'val_label.npy',label)
        #Save the images
        for i in range(data.shape[0]):
            cv2.imwrite(self.data_path+self.val_data_file+'val'+str(i+1)+'.png',data[i,:,:])
        print 'Generating validating data has been completed'
    #Remove all the training data and valiadting data
    def remove(self):
        shutil.rmtree(self.data_path+self.train_data_file)
        shutil.rmtree(self.data_path+self.train_label_file)
        shutil.rmtree(self.data_path+self.val_data_file)
        shutil.rmtree(self.data_path+self.val_label_file)
        print 'The data for training and validating has been removed'

if __name__=='__main__':
    a=prepare_mnist()
    #a.remove()
    a.generate_train()
    a.generate_val()
