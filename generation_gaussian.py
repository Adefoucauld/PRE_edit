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

np.random.seed(1)


def f(x):
    """The function to predict."""
    return x * np.sin(x)

# ----------------------------------------------------------------------
# # #  First the noiseless case
# X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

# # # # Observations
# y = f(X).ravel()

# # # # Mesh the input space for evaluations of the real function, the prediction and
# # # # its MSE
# x = np.atleast_2d(np.linspace(0, 10, 1000)).T

# #plt.figure()
# #plt.plot(x, f(x), 'b:', label=r'$f(x) = x\,\sin(x)$')
# #plt.plot(X, y, 'r.', markersize=10, label='Observations')

#-----------------------------------------------------------------------#
# Noisy case
X = np.linspace(0.1, 9.9, 20)
X = np.atleast_2d(X).T
#mesh the input spacefor evaluation of the real function
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

# Observations and noise
y = f(X).ravel()
dy = 0.5 + 1.0 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise

#test génération de data
# on va créer un X matrix avec un ensemble de 6 points par ligne, créés aléatoirement
#notre y va être la matrice correspondant à ces différents ensemble de points.


#on choisit de prendre 10 sets de data

m =20 #no of data set
n = 6 #no of points of interest to approx the gaussian line

X_bis = np.zeros((m,n),dtype = float)

for i in range(m):
    X_bis[i,:]= np.random.random(n)*10
    
#on crée nos différents sets de data ( que l'on 
# aurait pu sélectionner de manièrer aléatoire )

X_train = X_bis[:12,:]
X_val = X_bis[12:16,:]
X_test = X_bis[16:,:]

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
# for i in range(6):
#     plt.plot(X_train[i,:],y_train[i,:],linestyle = 'none',marker ='o')


# ## Noisy case
plt.figure()
plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')
for i in range(3):
    plt.plot(X_train[i,:],y_train[i,:],linestyle = 'none',marker ='o',color= (i/4,i/5,0.8-i/4))
    plt.errorbar(X_train[i,:].ravel(), y_train[i,:], dy_train[i,:],fmt='none',color=(i/4,i/5,0.8-i/4))

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

training_set  = MyDataset(X_train,y_train) # on a chargé nos données

train_loading = torch.utils.data.DataLoader(training_set, batch_size= 20)



# Ecriture du réseau de neurones (reprise du tp_deep)



class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.FC1 = nn.Linear(6,3)
    self.FC2 = nn.Linear(3, 6)
  def forward(self, x):
    #x = F.sigmoid(self.FC1(x)) # j'ai du le remplacer car F.sigmoid était indiqué " deprecated"
    x = torch.sigmoid(self.FC1(x))
    x = self.FC2(x)
    return x

model = Net()

#entrainement et validation

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
lr=0.01, weight_decay= 1e-3, momentum = 0.9)



def train(net, train_loader, optimizer, epoch):
    net.train()
    total_loss=0
    for batch_idx, (data, target) in enumerate(train_loader, 0):
        #data, target = data.to(device), target.to(device)
        #(complete the code (GF) )
        data,target=torch.FloatTensor(data.float()).view(-1,1),torch.FloatTensor(target.float())
        optimizer.zero_grad()
        outputs = net(data)
        print('outputs',outputs)
        loss = criterion(outputs,target)
        loss.backward()
        total_loss +=loss.cpu().item()
        optimizer.step()
    torch.optim.lr_scheduler.step()
    print('Epoch:', epoch , 'average loss ', total_loss/ len(train_loader))


val_loading = MyDataset(X_val, y_val) # on charge données de validation

test_loading = MyDataset(X_test, y_test) #on charge données de test

def test(net,test_loader):
    total_loss = 0
    for batch_idx,(test_loader.data_feature, test_loader.data_target) in enumerate(test_loader,0):
        outputs = net(test_loader.data_feature)
        loss = criterion(outputs,test_loader.data_target)
        total_loss += loss.cpu().item()
    print('average loss', total_loss/len(test_loader))
    
        
#on a définit nos fonctions de train et de test, 
# on va maintenant les utiliser
    
for epoch in range(50):
    train(model,train_loading,optimizer,epoch)
    test(Net,test_loading)    
    
