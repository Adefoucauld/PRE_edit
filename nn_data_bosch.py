# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 18:17:51 2020

@author: Utilisateur
"""



import csv 
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt 

np.random.seed(1)



# "Pour fichier text"
data_file_path = 'C:/Users/Utilisateur/Documents/IA/V2O3_Pump_0.9mW_Probe_0.05mW8Step_1micron_210918_Adjusted.txt'
# data = pd.read_csv(data_file_path, header = None,index_col = None, sep ='\t')

# " pour fichier ADF"
# data2_file_path = 'C:/Users/Utilisateur/Documents/IA/V2O3_Pump_0.05mW_Probe_0.05mW_Step_1micron_26091800.ADF'
# data2 = pd.read_csv(data2_file_path,header = None, index_col = None , sep = '\t', skiprows = lambda x : x<= 9)



def conversion(L):      # permet le traitement de fichier csv comme une liste
    T=[]
    for i in L:
        T.append(float(i.replace(',','.')))
    return(T)
    
time,ampl,phase =[],[],[]
with open(data_file_path, 'r') as csvfile:  #ce programme va permettre de traiter tout enregistrement des deux capteurs
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        time.append(row[0])
        ampl.append(row[1])
        phase.append(row[2])
        
time=conversion(time)   
ampl=conversion(ampl)
phase = conversion(phase)


time_vec= np.array(time)
ampl_vec = np.array(ampl)
phase_vec = np.array(phase)

X_train = time_vec[:200]
X_val = time_vec[200:281]
X_test = time_vec[281:361]

y_train = ampl_vec[:200]
y_val = ampl_vec[200:281]
y_test = ampl_vec[281:361]

# calcul de mean_X_train et std_X_train, idem pour y
    

mean_X_train = np.mean(X_train)
std_X_train = np.std(X_train)

mean_y_train = np.mean(y_train)
std_y_train =np.std(y_train)




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
    
    
for epoch in range(50): 
    train(model,train_loading,optimizer,epoch)
    test(model,val_loading,loss_list_val)
    test_no_norm(model, val_loading,loss_list)
    test_no_norm(model,test_loading,loss_list_test)
print('Epoch:', epoch , 'average training loss ', loss_list_train[-1])
print( 'average testing loss ', loss_list_val[-1])
   

plt.figure(2)
plt.plot(loss_list_train,'r',label = 'Training loss')
# plt.plot(loss_list,'g',label = ' Validation loss')
# plt.plot(loss_list_test,'b',label = ' Testing loss')
# plt.legend()
      
