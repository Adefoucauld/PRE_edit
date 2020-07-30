# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 22:21:47 2020

@author: Utilisateur
"""

import csv 
import numpy as np
from matplotlib import pyplot as plt


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


plt.figure(1)
plt.plot(time,ampl,'r')

time_vec= np.array(time)
ampl_vec = np.array(ampl)
phase_vec = np.array(phase)


