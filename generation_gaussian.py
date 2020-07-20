# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:53:24 2020

@author: Utilisateur
"""


import numpy as np

from matplotlib import pyplot as plt

np.random.seed(1)


def f(x):
    """The function to predict."""
    return x * np.sin(x)

# ----------------------------------------------------------------------
# #  First the noiseless case
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

# # Observations
y = f(X).ravel()

# # Mesh the input space for evaluations of the real function, the prediction and
# # its MSE
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

# #plt.figure()
# #plt.plot(x, f(x), 'b:', label=r'$f(x) = x\,\sin(x)$')
# #plt.plot(X, y, 'r.', markersize=10, label='Observations')


#test génération de data
# on va créer un X matrix avec un ensemble de 6 points par ligne, créés aléatoirement
#notre y va être la matrice correspondant à ces différents ensemble de points.


#on choisit de prendre 10 sets de data

m =10 #no of data set
n = 6 #no of points of interest

X_bis = np.zeros((m,n),dtype = float)

for i in range(m):
    X_bis[i,:]= np.random.random(6)*10
    
#on crée nos différents sets de data ( que l'on 
# aurait pu sélectionner de manièrer aléatoire )

X_train = X_bis[:6,:]
X_val = X_bis[6:8,:]
X_test = X_bis[8:,:]

y_train = np.reshape(f(X_train).ravel(),[6,n])
y_val = np.reshape(f(X_val).ravel(),[2,n])

plt.figure()
plt.plot(x, f(x), 'b:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label='Observations')
for i in range(6):
    plt.plot(X_train[i,:],y_train[i,:],linestyle = 'none',marker ='o')

# calcul de mean_X_train et std_X_train, idem pour y
    
mean_X_train = np.mean(X_train)
std_X_train = np.std(X_train)

mean_y_train = np.mean(y_train)
std_y_train =np.std(y_train)

# Ecriture du data loader (repris du tp_deep)

import torch
from torch.utils import data

class MyDataset(data.Dataset):

#Characterizes a dataset for Pytorch
    def __init__(self, data_feature, data_target):
        #Initialization
        self.data_feature = data_feature
        self.data_target = data_target
        self.transformed_feature = self.transforms_feature()
        self.transformed_target = self.transforms_target()
        
    def __len__(self):
    #Denotes the total number of samples
        return len(self.data_feature)
    
    def __getitem__(self, index):
    #Generates one sample of data
    # Select sample
        data_feature = torch.from_numpy(self.transformed_feature[index]).float()
        data_target = torch.from_numpy(self.transformed_target[index]).float()
        return data_feature, data_target

    def transforms_feature(self ):
        return (self.data_feature - np.full(self.data_feature.shape,mean_X_train)) / np.full(self.data_feature.shape, std_X_train)
    
    def transforms_target(self ):
        y_train_normalized = (self.data_target - mean_y_train) / std_y_train
        return np.array(y_train_normalized, ndmin = 2).T

X  = MyDataset(X_train, y_train) # on a chargé nos données

# Ecriture du réseau de neurones (reprise du tp_deep)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.FC1 = nn.Linear(6, 3)
    self.FC2 = nn.Linear(3, 1)
  def forward(self, x):
    x = F.sigmoid(self.FC1(x))
    x = self.FC2(x)
    return x

model = Net()

#entrainement et validation

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
lr=0.01, weight_decay= 1e-3, momentum = 0.9)

# zeroes the gradient buffers of all parameters
optimizer.zero_grad()
outputs = model(X)
loss = criterion(outputs, y_train)
loss.backward()
# Perform the training parameters update
optimizer.step()   

