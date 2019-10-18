#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:14:45 2017
    
@author: hq
"""
import os
import glob
import numpy as np
import pandas as pd 
from time import time

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from scipy.stats import truncnorm
import matplotlib.pyplot as plt
###############################################################################
## additional function 
      
def loss_plot(losses, names, save = False):
    
    if save == True:
        plt.figure(figsize = (5,4))
        plt.plot(losses["d"], label='discriminitive loss')
        plt.plot(losses["g"], label='generative loss')
        plt.plot(losses["q"], label='classifier loss')
        plt.legend()
        plt.savefig(names)
    
    if save == False:
        plt.figure(figsize = (5,4))
        plt.plot(losses["d"], label='discriminitive loss')
        plt.plot(losses["g"], label='generative loss')
        plt.plot(losses["q"], label='classifier loss')
        plt.legend()
        plt.show()
        
        
############################################################################

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def log(x):
    return tf.log(x + 1e-8)
            
def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)

#############################################################################
    
## loss function for Cat GaN
def entropy1(p, batch_size,tol = 1e-8):
    y1 = tf.reduce_sum(p, axis = 0) / batch_size
    y2 = tf.reduce_sum(-y1 * tf.log(y1 + tol))
    return y2
        
def entropy2(p, batch_size, tol = 1e-8):
    y1 = -p * tf.log(p + tol)               
    y2 = tf.reduce_sum(y1) / batch_size
    return y2

def cross_entropy(p, Y, tol = 1e-8):
    y1 = tf.log(p + tol)
    y2 = tf.reduce_mean(tf.reduce_sum(Y * y1, 1))
    return y2

#############################################################################

def discriminator(inputs, label, shp = 170, clust = 10,
                  batch_size = 32, reuse = False):
    with tf.variable_scope('discriminator',reuse = reuse):
            
            cond = tf.layers.dense(label, shp, 
                                   kernel_initializer = tf.contrib.layers.xavier_initializer())
            cond = tf.reshape(cond, [batch_size, shp, 1])
             
            con_inputs = tf.concat([inputs, cond],axis=2)
            
        
            H = tf.layers.conv1d(con_inputs, 64, 10, strides = 2, padding = 'same',
                                 activation =None,
                                 kernel_initializer = tf.contrib.layers.xavier_initializer())       
            H = lrelu(H, 0.05)
            H = tf.layers.dropout(H, 0.5)
                
            H = tf.layers.conv1d(H, 128, 5, strides = 2, padding = 'same' ,
                                 activation =None,
                                 kernel_initializer = tf.contrib.layers.xavier_initializer())       
            H = lrelu(H, 0.05)
            H = tf.layers.dropout(H, 0.5)
                       
            H = tf.layers.conv1d(H, 256, 5, strides = 2, padding = 'same' ,
                                 activation =None,
                                 kernel_initializer = tf.contrib.layers.xavier_initializer())       
            H = lrelu(H, 0.05)
            H = tf.layers.dropout(H, 0.5)
            
            '''
            H = tf.layers.conv1d(H, 512, 3, strides = 2, padding = 'same' ,
                                 activation =None,
                                 kernel_initializer = tf.contrib.layers.xavier_initializer())       
            H = lrelu(H, 0.05)
            H = tf.layers.dropout(H, 0.5)
            '''
        
            H1 = tf.contrib.layers.flatten(H)
                
            output_logit = tf.layers.dense(H1, clust, activation = None) ## Catgan loss      
            output = tf.nn.softmax(output_logit)
            
    return output, output_logit,  H

def generator(noise, label_class, label_add, shp = 170, reuse=False):
    with tf.variable_scope('generator',reuse = reuse):

            H = tf.concat([noise, tf.concat([label_class, label_add],1)], 1)
        
            H = tf.layers.dense(H, int(shp)*8,
                                kernel_initializer = tf.contrib.layers.xavier_initializer())        
            H = tf.reshape(H, [-1, int(shp)*8, 1])
            H = tf.layers.batch_normalization(H)
            H = lrelu(H, 0.05)

            H = tf.layers.conv1d(H, 256, 3, strides = 2, padding = 'same'  ,
                                 activation = None,
                                 kernel_initializer = tf.contrib.layers.xavier_initializer())       
            H = tf.layers.batch_normalization(H)        
            H = lrelu(H, 0.05)

            
            
            H = tf.layers.conv1d(H, 128, 5, strides = 2, padding = 'same'  ,
                                 activation = None,
                                 kernel_initializer = tf.contrib.layers.xavier_initializer())       
            H = tf.layers.batch_normalization(H)        
            H = lrelu(H, 0.05)
            
        
            H = tf.layers.conv1d(H, 64, 5, strides = 2, padding = 'same'  ,
                                 activation = None,
                                 kernel_initializer = tf.contrib.layers.xavier_initializer())       
            H = tf.layers.batch_normalization(H)        
            H = lrelu(H, 0.05)

            
            output = tf.layers.conv1d(H, 1, 10, padding = 'same'  , activation = tf.nn.sigmoid)      
                
    return output

def classifier(layer, classes, reuse = False):
    with tf.variable_scope("classifier", reuse = reuse):
        
        net = tf.contrib.layers.flatten(layer)
                
        net = tf.layers.dense(net, 128, 
                              activation = None,
                              kernel_initializer = tf.contrib.layers.xavier_initializer())
        net = tf.layers.batch_normalization(net)
        net = lrelu(net, 0.05)

        out_logit = tf.layers.dense(net, classes)
        out = tf.nn.softmax(out_logit)
        
        return out, out_logit
        
        
###############################################################################
### read file in dir
