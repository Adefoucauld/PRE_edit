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
#  First the noiseless case
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

# Observations
y = f(X).ravel()

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

plt.figure()
plt.plot(x, f(x), 'b:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label='Observations')


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


