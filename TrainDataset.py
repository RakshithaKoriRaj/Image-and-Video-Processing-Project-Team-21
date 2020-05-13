#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 00:32:18 2020

"""
import os
import cv2
import math
import numpy as np
from tqdm import tqdm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch 
import Config
import time
import plotmodel


REBUILD_DATA = True
TRAINING_PER_LABEL = 15300

GLOBAL_IMAGE_SIZE = int(Config.DOWNSAMPLE_SIZE[0] * Config.CROP_SCALING)
GLOBAL_MAX_POOL_SIZE = 2 
LEARNING_RATE = 0.001

MAX_PER_LABEL = 400
RANDOMIZE_LABELS = False
SHOW_IMAGES = False

BATCH_SIZE = 100
EPOCHS = 10

VAL_PCT = 0.1

COVID = os.path.join(os.path.abspath(os.getcwd()),'datasets','augmented-dataset','covid')
NORMAL = os.path.join(os.path.abspath(os.getcwd()),'datasets','augmented-dataset','normal')
LABELS = {COVID:0,NORMAL:1}

LABELS_IDX = ['COVID','NORMAL']

class CovidVsNormal():
    IMG_SIZE = GLOBAL_IMAGE_SIZE
    
    training_data = []
    test_data = []
    train_data = []
    covidcount = 0
    normalcount = 0 
    
    def make_test_data(self):
       
        for label in LABELS:
            print(label)
            
            files = os.listdir(label+"_test")
            np.random.shuffle(files)
            files = files[:TRAINING_PER_LABEL]
            for f in tqdm(files):
                path = os.path.join(label+"_test",f)
                img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE))
                self.test_data.append([np.array(img).astype(float),np.eye(2)[LABELS[label]]])
                if label == COVID:
                    self.covidcount += 1
                elif label == NORMAL:
                    self.normalcount +=1
        np.save("test_data.npy",self.test_data)
        print("Covid:",self.covidcount)
        print("Normal:",self.normalcount)
    
    def make_train_data(self):
        for label in LABELS:
            print(label)
            
            files = os.listdir(label+"_train")
            np.random.shuffle(files)
            files = files[:TRAINING_PER_LABEL]
            for f in tqdm(files):
                path = os.path.join(label+"_train",f)
                img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE))
                self.train_data.append([np.array(img).astype(float),np.eye(2)[LABELS[label]]])
                if label == COVID:
                    self.covidcount += 1
                elif label == NORMAL:
                    self.normalcount +=1
        np.save("train_data.npy",self.train_data)
        print("Covid:",self.covidcount)
        print("Normal:",self.normalcount)

if REBUILD_DATA:
    cvn = CovidVsNormal()
    cvn.make_test_data()
    cvn.make_train_data()


train_data = np.load("train_data.npy",allow_pickle=True)

print(len(train_data))


test_data = np.load("test_data.npy",allow_pickle=True)

print(len(test_data))



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,8,3)
        self.conv2 = nn.Conv2d(8,16,5)
        self.conv3 = nn.Conv2d(16,32,5)
        #This is to get the size of fc . But can be calcualted
        x = torch.randn(GLOBAL_IMAGE_SIZE,GLOBAL_IMAGE_SIZE).view(-1,1,GLOBAL_IMAGE_SIZE,GLOBAL_IMAGE_SIZE)
        self._to_linear = None
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear,64)
        self.fc2 = nn.Linear(64,2)
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

def run_net(count, rate, batch_size, doRandom=False,allDataSet=True):
    def get_partial(data, count):
        partial = []
        numCovid = 0
        numNormal = 0
        covidLabel = np.eye(2)[LABELS[COVID]]
        normalLabel = np.eye(2)[LABELS[NORMAL]]
        for point in data:
            if doRandom:
                label = [covidLabel, normalLabel][np.random.randint(2)]
            else:
                label = point[1]
            if np.array_equal(label, covidLabel):
                if numCovid < count:
                    partial.append(point)
                    numCovid += 1
            elif np.array_equal(label, normalLabel):
                if numNormal < count:
                    partial.append(point)
                    numNormal += 1
            else:
                print("Unknown label!")
        print("{} covid".format(numCovid))
        print("{} normal".format(numNormal))
        print("{} total".format(len(partial)))
        return np.array(partial)
    
    def show_gradients(batch_X,batch_y,OUT,batch_number=0,epoch=0):
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10,5))
        for col in range(4):
            grad = batch_X.grad[col][0].cpu().abs().numpy()
            img = batch_X[col][0].cpu().detach().numpy()
            axes[0][col].axis('off')
            axes[1][col].axis('off')
            axes[0][col].imshow(img, cmap='gray')
            axes[0][col].set_title("Actual:{}".format(LABELS_IDX[torch.argmax(batch_y[col])]))
            axes[1][col].imshow(grad, cmap=plt.cm.hot)
            axes[1][col].set_title("Predicted:{}".format(LABELS_IDX[torch.argmax(OUT[col])]))
            
        fig.suptitle("{0} batch size {1} : batch_number {2} : epoch {3}".format(count,doRandom,batch_number,epoch ))
        if SHOW_IMAGES:
            plt.show(block=False)
        plt.savefig("gradient_images/{3}e{2}bn{0}bs{1}ran.png".format(count,doRandom,batch_number,epoch ))
        plt.close()
    
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("running on the GPU")
    else:
        device = torch.device("cpu")
        print("running on cpu")

    def randomize_labels(dataset):
        covidLabel = np.eye(2)[LABELS[COVID]]
        normalLabel = np.eye(2)[LABELS[NORMAL]]
        return [[covidLabel, normalLabel][np.random.randint(2)] for data in dataset]

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    
    X_train = [i[0]/255.0 for i in train_data]
    if doRandom:
        y_train = randomize_labels(train_data)
    else:
        y_train = [i[1] for i in train_data]

    X_test = [i[0]/255.0 for i in test_data]
    if doRandom:
        y_test = randomize_labels(test_data)
    else:
        y_test = [i[1] for i in test_data]


    train_X = X_train
    train_y = y_train
    test_X = torch.Tensor(X_test).view(-1,GLOBAL_IMAGE_SIZE,GLOBAL_IMAGE_SIZE)
    test_y = torch.Tensor(y_test)
    print("train X length:",len(train_X))
    print("test X length:",len(test_X)) 
    
    def fwd_pass(X,y,train = False):
        if train:
            net.zero_grad()
        outputs = net(X)
        matches = [torch.argmax(i)==torch.argmax(j) for i,j in zip(outputs,y)]
        acc = matches.count(True)/ len(matches)
        loss = loss_fuction(outputs,y)
        if train:
            loss.backward(retain_graph=True)
            optimizer.step()
        return acc , loss , outputs
            
    def test(size=16):
        random_start = np.random.randint(len(test_y)-size)
        X , y = test_X[random_start:random_start+size] , test_y[random_start:random_start+size]
        with torch.no_grad():
            val_acc, val_loss,_ = fwd_pass(X.float().view(-1,1,GLOBAL_IMAGE_SIZE,GLOBAL_IMAGE_SIZE).to(device),y.to(device))
        return val_acc, val_loss
    
    MODEL_NAME = "model-E{0}-B{1}-R{2}-tr{3}-ts{4}-{5}".format(EPOCHS, batch_size, doRandom,len(train_y),len(test_y),int(time.time()))
    print(MODEL_NAME)
    net = Net().to(device)

    #used for back propogation
    optimizer = optim.Adam(net.parameters(),lr=rate)
    loss_fuction = nn.MSELoss()

    
    def train():
        with open("model.log","a") as f:
            for epoch in range(EPOCHS):
                for i in tqdm(range(0,len(train_X),batch_size)):
                    batch_X = torch.tensor(train_X[i:i+batch_size]).float().view(-1,1,GLOBAL_IMAGE_SIZE,GLOBAL_IMAGE_SIZE)
                    batch_y = torch.Tensor(train_y[i:i+batch_size])
                    batch_X , batch_y = batch_X.to(device) , batch_y.to(device)
                    batch_X.requires_grad_(True)
                    acc , loss, outputs = fwd_pass(batch_X,batch_y,train = True)
                    val_acc , val_loss = test(20)
                    name = "{0},{1},{2},{3},{4},{5}\n".format(
                        MODEL_NAME,
                        round(time.time(),3),
                        round(float(acc),5),
                        round(float(loss),5),
                        round(float(val_acc),5),
                        round(float(val_loss),5))
                    f.write(name)
                    if (i/batch_size)%5==0:
                        show_gradients(batch_X,batch_y,outputs,i/batch_size,epoch)
            show_gradients(batch_X,batch_y,outputs,i/batch_size,epoch)
            torch.save(net.state_dict(), "models/{}.pt".format(MODEL_NAME))
        return acc , loss , MODEL_NAME
    
    
    return train()
    
        


    
def complete_run():
    print("BATCH_SIZE:"+str(BATCH_SIZE))
    accuracy, loss,model_name = run_net(0, LEARNING_RATE, BATCH_SIZE, False, True)
    plotmodel.create_acc_loss_graph(model_name)

print("Timing {}".format(complete_run))
start = time.time()
complete_run()
end = time.time()
print("{0} took {1:.2f} seconds".format('complete', end - start))



