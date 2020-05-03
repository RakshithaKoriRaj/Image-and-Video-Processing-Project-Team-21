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
import Config

REBUILD_DATA = False
GLOBAL_IMAGE_SIZE = int(Config.DOWNSAMPLE_SIZE[0] * Config.CROP_SCALING)
GLOBAL_MAX_POOL_SIZE = 2 
LEARNING_RATE = 0.001
MAX_PER_LABEL = 800

RANDOMIZE_LABELS = False

BATCH_SIZE = 100
EPOCHS = 2

VAL_PCT = 0.1

COVID = os.path.join(os.path.abspath(os.getcwd()),'datasets','augmented-dataset','covid')
NORMAL = os.path.join(os.path.abspath(os.getcwd()),'datasets','augmented-dataset','normal')
LABELS = {COVID:0,NORMAL:1}

class CovidVsNormal():
    IMG_SIZE = GLOBAL_IMAGE_SIZE
    
    training_data = []
    covidcount = 0
    normalcount = 0 
    
    def make_training_data(self):
        for label in LABELS:
            print(label)
            files = os.listdir(label)
            for f in tqdm(files):
                path = os.path.join(label,f)
                img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE))
                self.training_data.append([np.array(img),np.eye(2)[LABELS[label]]])
                if label == COVID:
                    self.covidcount += 1
                elif label == NORMAL:
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

def run_net(count, rate, batch_size, doRandom=False):
	net = Net()

	#used for back propogation
	optimizer = optim.Adam(net.parameters(),lr=rate)
	loss_fuction = nn.MSELoss()

	np.random.shuffle(training_data)

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

	partial_data = get_partial(training_data, count)
	'''
	image_shape = partial_data[0][0].shape
	width = image_shape[0]
	height = image_shape[1]
	'''

	X = torch.Tensor([i[0] for i in partial_data]).view(-1,GLOBAL_IMAGE_SIZE,GLOBAL_IMAGE_SIZE)
	X = X/255.0 
	y = torch.Tensor([i[1] for i in partial_data])

	val_size = int(len(X)*VAL_PCT)
	print("Val size:",val_size)

	train_X = X[:-val_size]
	train_y = y[:-val_size]
	test_X = X[-val_size:]
	test_y = y[-val_size:]
	print("train X length:",len(train_X))
	print("text X length:",len(test_X))

	for epoch in range(EPOCHS):
	    for i in tqdm(range(0,len(train_X),batch_size)):
	        batch_X = train_X[i:i+batch_size].view(-1,1,GLOBAL_IMAGE_SIZE,GLOBAL_IMAGE_SIZE)
	        batch_y = train_y[i:i+batch_size]
	        net.zero_grad()
	        outputs = net(batch_X)
	        loss = loss_fuction(outputs,batch_y)
	        loss.backward()
	        optimizer.step()

	torch.save(net.state_dict(), "models/model-{}.pt".format(count))
	print("Loss:",loss)

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

	accuracy = round(correct/total,3)
	print("Accuracy:",accuracy)
	return accuracy, loss

def tests(num, doRandom):
	counts = [(2 ** x) for x in range(num)]
	pairs = []
	for count in counts:
		samples = count * 100
		rate = 0.0015 / count
		batch_size = int(samples / 10)
		accuracy, loss = run_net(samples, rate, batch_size, doRandom)
		pairs.append([samples, rate, accuracy, loss, batch_size])
	return pairs

p1 = tests(6, True)
p2 = tests(6, False)

def print_result(result):
	print("\tSamples: {0}\n\t\tLearning rate: {1}\n\t\tAccuracy: {2:.2f}\n\t\tLoss: {3}\n\t\tBatchSize: {4}".format(*result))

print("With random labels" + "-"*10)
for result in p1:
	print_result(result)
print("With matching labels" + "-"*10)
for result in p2:
	print_result(result)