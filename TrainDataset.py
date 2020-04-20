#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 00:32:18 2020

"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch 

REBUILD_DATA = False
GLOBAL_IMAGE_SIZE = 100
GLOBAL_MAX_POOL_SIZE = 2 
LEARNING_RATE = 0.001

class CovidVsNormal():
    IMG_SIZE = GLOBAL_IMAGE_SIZE
    COVID = os.path.join(os.path.abspath(os.getcwd()),'datasets','augmented-dataset','covid')
    NORMAL = os.path.join(os.path.abspath(os.getcwd()),'datasets','augmented-dataset','normal')
    LABELS = {COVID:0,NORMAL:1}
    training_data = []
    covidcount = 0
    normalcount = 0 
    
    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                path = os.path.join(label,f)
                img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE))
                self.training_data.append([np.array(img),np.eye(2)[self.LABELS[label]]])
                if label == self.COVID:
                    self.covidcount += 1
                elif label == self.NORMAL:
                    self.normalcount +=1
        np.random.shuffle(self.training_data)
        np.save("training_data.npy",self.training_data)
        print("Covid:",self.covidcount)
        print("Normal:",self.normalcount)

if REBUILD_DATA:
    cvn = CovidVsNormal()
    cvn.make_training_data()

training_data = np.load("training_data.npy",allow_pickle=True)

print(len(training_data))
#plt.imshow(training_data[1][0],cmap="gray")
#plt.show()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,5)
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5)
        #This is to get the size of fc . But can be calcualted
        x = torch.randn(GLOBAL_IMAGE_SIZE,GLOBAL_IMAGE_SIZE).view(-1,1,GLOBAL_IMAGE_SIZE,GLOBAL_IMAGE_SIZE)
        self._to_linear = None
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear,512)
        self.fc2 = nn.Linear(512,2)
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(GLOBAL_MAX_POOL_SIZE,GLOBAL_MAX_POOL_SIZE))
        x = F.max_pool2d(F.relu(self.conv2(x)),(GLOBAL_MAX_POOL_SIZE,GLOBAL_MAX_POOL_SIZE))
        x = F.max_pool2d(F.relu(self.conv3(x)),(GLOBAL_MAX_POOL_SIZE,GLOBAL_MAX_POOL_SIZE))
        #print(x[0].shape)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    
    def forward(self,x):
        x = self.convs(x)
        x = x.view(-1,self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1 )

net = Net()

#used for back propogation
optimizer = optim.Adam(net.parameters(),lr=0.001)
loss_fuction = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1,GLOBAL_IMAGE_SIZE,GLOBAL_IMAGE_SIZE)
X = X/255.0 
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]
test_X = X[-val_size:]
test_y = y[-val_size:]
print(len(train_X))
print(len(test_X))

BATCH_SIZE = 100
EPOCHS = 1

for epoch in range(EPOCHS):
    for i in tqdm(range(0,len(train_X),BATCH_SIZE)):
        batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,GLOBAL_IMAGE_SIZE,GLOBAL_IMAGE_SIZE)
        batch_y = train_y[i:i+BATCH_SIZE]
        net.zero_grad()
        outputs = net(batch_X)
        loss = loss_fuction(outputs,batch_y)
        loss.backward()
        optimizer.step()

print(loss)

correct = 0 
total = 0 
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1,1,GLOBAL_IMAGE_SIZE,GLOBAL_IMAGE_SIZE))[0]
        predicted_class = torch.argmax(net_out)
        #print(real_class,net_out)
        if real_class == predicted_class:
            correct +=1
        total+=1

print("Accuracy:",round(correct/total,3))
