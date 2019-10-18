# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 15:31:24 2018

@author: HQ
"""

import os,sys
import glob
import numpy as np
import pandas as pd 
from time import time

def loadDat(file):
    ## input: csv file 
    ## output: np.array / dtype: float32
    
    train = pd.read_csv(file)
    test = pd.read_csv(file.replace("TRAIN","TEST"))

    train = train.drop(train.columns[0], axis=1)
    test = test.drop(test.columns[0], axis=1)
    
    col_idx_tr = train.columns.get_loc('label')
    col_idx_te = test.columns.get_loc('label') 
    
    x_train = train.iloc[:,(col_idx_tr+1):]
    y_train = train.iloc[:,col_idx_tr]
    x_test = test.iloc[:,(col_idx_te+1):]
    y_test = test.iloc[:,col_idx_te]    
    
    
    print('data name: ', file[file.find('\\')+1:])
    print('data (train) :',x_train.shape[0],"| (test) :",x_test.shape[0])
    print('time step :', x_train.shape[1],
          "| imbalance ratio : ", round(sum(y_train==-1)/sum(y_train==1),2))
    
    
    return x_train, y_train, x_test, y_test 


def loOP(train, extent = 2, n = 20):
    from PyNomaly import loop
    ## input: train data 
    ## output: train column + 1 (LocalOutlierProbability)
    prob = loop.LocalOutlierProbability(train, extent=extent, 
                                        n_neighbors=n).fit()
    scores = prob.local_outlier_probabilities.reshape(train.shape[0],1)
        
    return scores

def probNN(x_train, y_train, n = 20):
    from sklearn.neighbors import NearestNeighbors
    ## input: train
    ## output: the # of same class obs / K
    ## type: e='euclideanDist', m='mantahhanDist', d='dynamicTimeWarrap'
    nbrs = NearestNeighbors(n_neighbors=n+1, algorithm='auto').fit(x_train)
    _, idx = nbrs.kneighbors(x_train)
    idx = idx[:,1:]
    
    scores = []
    for i in range(len(y_train)):
        if y_train[i] == 1:
            ## minority
            scores.append(np.sum(y_train[idx[i]]==1)/n)
        else:
            ## majority
            scores.append(np.sum(y_train[idx[i]]==0)/n)
    
    scores = np.array(scores,dtype='float32').reshape(x_train.shape[0],1)
        
    return scores

def merge(prob_lof, prob_nn, weight = 0.5):
    
    dat_info = (weight * prob_nn + (1 - weight) * (1-prob_lof))
    
    return dat_info


######################################################################
## PCA plot  
'''    
import matplotlib.pyplot as plt
import matplotlib.colors

pca = PCA(n_components=2)

X_pca = pca.fit(x_train).transform(x_train)
y_train[y_train==-1] = y_train[y_train==-1] + 1

#plt.figure(figsize=(6,6))
color = ["blue","red"]
target_names = ["majority","minority"]
lw = 3
for color, i, target_name in zip(color, [0, 1], target_names):
    plt.scatter(X_pca[y_train == i,0], X_pca[y_train == i, 1], color=color, alpha=.4, lw=lw,
                label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA')       
    
#######################################################################
score = loOP(x_train,0.997,10)    
score = score.astype(float).reshape(score.size)
## 1 = 0.68
## 2 = 0.95 
## 3 = 0.997

plt.scatter(X_pca[:,0], X_pca[:,1], c = score)
plt.colorbar()

plt.scatter(X_pca[:,0], X_pca[:,1], c = 1-score)
plt.colorbar()

###############################################################################    

score_maj, score_min = probNN_V2(x_train, y_train, n = 10)
score_min = score_min.astype(float).reshape(score_min.size)
plt.scatter(X_pca[:,0], X_pca[:,1], c= score_min)
plt.colorbar()


score_NN = probNN(x_train, y_train, n= 10)
score_NN = score_NN.astype(float).reshape(score_NN.size)
plt.scatter(X_pca[:,0], X_pca[:,1], c= score_NN)
plt.colorbar()

plt.scatter(X_pca[:,0], X_pca[:,1], c = score_min)
plt.colorbar()

plt.scatter(X_pca[:,0], X_pca[:,1], c = 1-nn_score, cmap = "gray_r")
plt.colorbar()

###############################################################################    
weight = .5
merge_s1 = (weight * score_min + (1 - weight) * (score))
plt.scatter(X_pca[:,0], X_pca[:,1], c = merge_s1)
plt.colorbar()

plt.scatter(X_pca[:,0], X_pca[:,1], c = merge_s1)
plt.colorbar()

merge_s2 = (weight * score_NN + (1 - weight) * (1-score))
plt.scatter(X_pca[y_train==1,0], X_pca[y_train==1,1], c = merge_s2[y_train==1])
plt.colorbar()


'''
    


