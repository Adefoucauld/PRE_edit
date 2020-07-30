# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:22:40 2020

@author: Utilisateur
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:53:24 2020

@author: Utilisateur
"""
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt 

np.random.seed(1)


def f(x):
    """The function to predict."""
    return x * np.sin(x)

# # Noisy case
X = np.linspace(0.1, 9.9, 20)
X = np.atleast_2d(X).T
#mesh the input spacefor evaluation of the real function
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

# # Observations and noise
y = f(X).ravel()
dy = 0.5 + 1.0 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise

#test génération de data

m =2000 #size of dataset

X_bis = np.zeros((1,m),dtype = float)

X_bis=np.random.random(m)*10
    
#on crée nos différents sets de data ( que l'on 
# aurait pu sélectionner de manièrer aléatoire )

X_train = X_bis[0:1200]
X_val = X_bis[1200:1600]
X_test = X_bis[1600:]

y_train = f(X_train) #on va bruiter les données pour chaque set de données
dy_train = 0.5 + 1.0 * np.random.random(y_train.shape)
noise_train = np.random.normal(0, dy_train)
y_train += noise_train

y_val = f(X_val)
dy_val = 0.5 + 1.0 * np.random.random(y_val.shape)
noise_val = np.random.normal(0, dy_val)
y_val += noise_val

y_test = f(X_test)

# Without noise
# plt.figure()
# plt.plot(x, f(x), 'b:', label=r'$f(x) = x\,\sin(x)$')
# plt.plot(X, y, 'r.', markersize=10, label='Observations')
# plt.plot(X_train,y_train,'g',marker='o',linestyle='none')

# # ## Noisy case
# plt.figure()
# plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
# plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')
# plt.legend()
# # for i in range(3):
# #     plt.plot(X_train[i,:],y_train[i,:],linestyle = 'none',marker ='o',color= (i/4,i/5,0.8-i/4))
# #     plt.errorbar(X_train[i,:].ravel(), y_train[i,:], dy_train[i,:],fmt='none',color=(i/4,i/5,0.8-i/4))

# calcul de mean_X_train et std_X_train, idem pour y
    

mean_X_train = np.mean(X_train)
std_X_train = np.std(X_train)

mean_y_train = np.mean(y_train)
std_y_train =np.std(y_train)

# Ecriture du data loader (repris du tp_deep)



class MyDataset(data.Dataset):

#Characterizes a dataset for Pytorch
    def __init__(self, data_feature, data_target):
        #Initialization
        self.data_feature = data_feature
        self.data_target = data_target
        # self.transformed_feature = self.transforms_feature()
        # self.transformed_target = self.transforms_target()
        
    def __len__(self):
    #Denotes the total number of samples
        return len(self.data_feature)
    
    def __getitem__(self, index):
    #Generates one sample of data
    # Select sample
        # data_feature = torch.from_numpy(self.transformed_feature[index]).float()
        # data_target = torch.from_numpy(self.transformed_target[index]).float()
        # return data_feature, data_target
        X_train_normalized = (self.data_feature[index] - mean_X_train) / std_X_train
        y_train_normalized = (self.data_target[index] - mean_y_train) / std_y_train
        return torch.from_numpy(np.array(X_train_normalized,ndmin=1)).float(), torch.from_numpy(np.array(y_train_normalized, ndmin = 1)).float()
                    
    # def transforms_feature(self ):
    #      X_train_transformed =(self.data_feature - np.full(self.data_feature.shape,mean_X_train)) / np.full(self.data_feature.shape, std_X_train)
    #      return torch.from_numpy(np.array(X_train_transformed,ndmin=1)).float()
     
    # def transforms_target(self ):
    #     y_train_transformed = (self.data_target - mean_y_train) / std_y_train
    #     return torch.from_numpy(np.array(y_train_transformed,ndmin=1)).float()

training_set  = MyDataset(X_train,y_train) # on charge nos données
train_loading = torch.utils.data.DataLoader(training_set, batch_size= 100)
    
val_set = MyDataset(X_val, y_val)  
val_loading = torch.utils.data.DataLoader(val_set, batch_size= 100)
    
test_set  = MyDataset(X_test,y_test) 
test_loading = torch.utils.data.DataLoader(test_set, batch_size= 100)


# Ecriture du réseau de neurones (reprise du tp_deep)


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.FC1 = nn.Linear(1,6)
    self.FC2 = nn.Linear(6, 1)
  def forward(self, x):
    x = F.relu(self.FC1(x)) 
    x = self.FC2(x)
    return x

model = Net()

#entrainement et validation

criterion = nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(),lr=0.0001, weight_decay= 0.001, momentum = 0.9)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.03,
                             weight_decay = 0.001) 

loss_list_train = []
loss_list_val = []
loss_list= []
loss_list_test = []

def train(net, train_loader, optimizer, epoch):
    net.train()
    total_loss=0
    for idx,(data, target) in enumerate(train_loader, 0):
        #data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs,target)
        loss.backward()
        total_loss +=loss.cpu().item()
        optimizer.step()
    loss_list_train.append(total_loss/len(train_loader))
    #torch.optim.lr_scheduler.step()
    #print('Epoch:', epoch , 'average training loss ', total_loss/ len(train_loader))


def test(net,test_loader,L):
    net.eval()
    total_loss = 0
    for idx,(data, target) in enumerate(test_loader,0):
        outputs = net(data)
        outputs = outputs * std_X_train + mean_X_train
        target = target * std_y_train + mean_y_train
        loss = criterion(outputs,target)
        total_loss += sqrt(loss.cpu().item())
    L.append(total_loss/len(test_loader))
    #print('average testing loss', total_loss/len(test_loader))
    
def test_no_norm(net,test_loader,L):
    net.eval()
    total_loss = 0
    for idx,(data, target) in enumerate(test_loader,0):
        outputs = net(data)
        loss = criterion(outputs,target)
        total_loss += sqrt(loss.cpu().item())
    L.append(total_loss/len(test_loader))
    #print('average testing loss', total_loss/len(test_loader))
    
        
#on a définit nos fonctions de train et de test, 
# on va maintenant les utiliser
    
for epoch in range(50): 
    train(model,train_loading,optimizer,epoch)
    test(model,val_loading,loss_list_val)
    test_no_norm(model, val_loading,loss_list)
    test_no_norm(model,test_loading,loss_list_test)
print('Epoch:', epoch , 'average training loss ', loss_list_train[-1])
print( 'average testing loss ', loss_list_val[-1])
   

plt.figure(2)
plt.plot(loss_list_train,'r',label = 'Training loss')
plt.plot(loss_list,'g',label = ' Validation loss')
plt.plot(loss_list_test,'b',label = ' Testing loss')
plt.legend()
      
