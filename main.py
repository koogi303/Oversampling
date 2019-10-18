# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:24:21 2018

@author: HQ
"""

import os,sys
import random
import glob
import numpy as np
import pandas as pd 
from time import time

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
     
#from scipy.stats import truncnorm
import matplotlib.pyplot as plt

from models import loaddat, proposed_bfgan

##############################

def sample_plot(plot_names, csv_names, save = False):
    
    if save == False:
        sample_size = int((sum(y_train[:,0]==1) - sum(y_train[:,0]==0)))
        if sample_size % 2 == 1:
            sample_size = sample_size - 1
                    
        gen_noise = np.random.normal(0,1,size=(sample_size,noise))
        gen_class = np.zeros([sample_size, 4])
        gen_class[:,1] = 1
        gen_class[:,3] = 1
                            
        
        generated_samples = sess.run(fake,
                                     feed_dict={Y_class: gen_class[:,0:2],
                                                Y_add: gen_class[:,2:4], 
                                                Z: gen_noise})
        generated_samples = np.squeeze(generated_samples, axis =2)
                    
        tmp_class = y_train[:,1]
        fake_minority = [2]*sum(gen_class[:,1]==1)
        tmp_class = np.concatenate((tmp_class,fake_minority))
                            
        pca = PCA(n_components=2)
        X_plot = np.squeeze(X_train, axis= 2)
        X_r = pca.fit(X_plot).transform(X_plot)
        X_fake = pca.fit(X_plot).transform(generated_samples)
                    
        X_r = np.concatenate((X_r, X_fake),axis=0)
                
        plt.figure(figsize=(6,6))
        color = ["red","blue","orange","purple"]
        target_names = ["majority","minority","fake_minority","fake_majority"]
        lw = 3
        for color, i, target_name in zip(color, [0, 1, 2, 3], target_names):
            plt.scatter(X_r[tmp_class == i, 0], X_r[tmp_class == i, 1], color=color, alpha=.4, lw=lw,
                        label=target_name)
            plt.legend(loc='best', shadow=False, scatterpoints=1)
            plt.title('PCA')       
   
    if save == True:
        sample_size = int((sum(y_train[:,0]==1) - sum(y_train[:,0]==0)))
        if sample_size % 2 == 1:
            sample_size = sample_size - 1
                    
        gen_noise = np.random.normal(0,1,size=(sample_size,noise))
        gen_class = np.zeros([sample_size, 4])
        gen_class[:,1] = 1
        gen_class[:,3] = 1

                            
        generated_samples = sess.run(fake,
                                     feed_dict={Y_class: gen_class[:,0:2],
                                                Y_add: gen_class[:,2:4], 
                                             Z: gen_noise})

        generated_samples = np.squeeze(generated_samples, axis =2)
        
        tmp_class = y_train[:,1]
        fake_minority = [2]*sum(gen_class[:,1]==1)
        tmp_class = np.concatenate((tmp_class,fake_minority))
        
                            
        pca = PCA(n_components=2)
        X_plot = np.squeeze(X_train, axis= 2)
        
               
        oversampled_X = pd.DataFrame(np.concatenate((X_plot, generated_samples),axis=0),
                                     columns = x_train.columns )
        oversampled_Y = pd.DataFrame(np.concatenate((y_train[:,1], np.array([1] * sum(gen_class[:,1]==1), dtype='float32')),axis=0),
                                     columns = ['label'])
        oversampled_dat = pd.concat([oversampled_Y,oversampled_X], axis=1)
        oversampled_dat.to_csv(csv_names)
        
        X_r = pca.fit(X_plot).transform(X_plot)
        X_fake = pca.fit(X_plot).transform(generated_samples)
                    
        X_r = np.concatenate((X_r, X_fake),axis=0)
                    
        plt.figure(figsize=(6,6))
        color = ["red","blue","orange","purple"]
        target_names = ["majority","minority","fake_minority","fake_majority"]
        lw = 3
        for color, i, target_name in zip(color, [0, 1, 2, 3], target_names):
            plt.scatter(X_r[tmp_class == i, 0], X_r[tmp_class == i, 1], color=color, alpha=.4, lw=lw,
                        label=target_name)
            plt.legend(loc='best', shadow=False, scatterpoints=1)
            plt.title('PCA')       
        plt.savefig(plot_names)    
                
###############################################
random.seed(1697)

##### preprocessing parameter 
neighbor = [3, 5, 7, 10, 20] 
weights = [0.,0.3,0.5,0.7,1.] ## lambda 

##### model parameter
margin = [0.1, 0.3 ,0.5, 0.7, 0.9] ## m
total_epoch = 500
batch_size = 32
        
     
path = './data'
spath = './save'

train_file = glob.glob(path + '/*_TRAIN.csv') 

for files in range(1,len(train_file)):
    for nn in neighbor:
        for w in weights:
            for m in margin:

                print("Neighbor :", nn, "| weights :", w, "| margins :", m)
                file = train_file[files-1] ## 
                
                x_train, y_train, x_test, y_test = loaddat.loadDat(file)
                prob_lof = loaddat.loOP(x_train, n = nn)
                prob_nn = loaddat.probNN(x_train, y_train, n= nn)
                
                #_, prob_nn = loaddat.probNN_V2(x_train, y_train, n = nn)
                #dat_info = loaddat.merge(prob_lof, prob_nn, weight = w)
                dat_info = loaddat.merge(prob_lof, prob_nn, weight = w)

                categorical_ = []
                for i in range(0,len(dat_info)):
                    if (m - dat_info[i]) >= 0:
                        categorical_.append(1) ## inner margin 
                    else:
                        categorical_.append(-1) ## outer margin
                #categorical_info = np.array(categorical_info, dtype='IN').reshape(dat_info.shape[0], 1)
                categorical_ = pd.DataFrame(categorical_)
                categorical_info = pd.get_dummies(categorical_.iloc[:,0]) ## out / in 

                if len(categorical_info.columns) == 2:
                    ##############################################################
                
                    y_train = pd.get_dummies(y_train)
                    
                    X_train = np.array(x_train, dtype= "float32")
                    y_train = np.array(y_train, dtype= "float32")
                    y_train = np.concatenate((y_train, categorical_info),axis=1)
                    
                    scaler = MinMaxScaler(feature_range=(0,1),copy=True)
                    scaler = scaler.fit(X_train)
                                    
                    X_dat = scaler.transform(X_train)
                    X_train = np.expand_dims(X_dat,axis=2)
                
                    #########
                    # haperparameter
                    ######
                    batch_size = batch_size
                    if batch_size % 2 == 1:
                        batch_size = batch_size + 1
                            
                    shp = X_train.shape[1]
                    noise = 100
                    classes = 2
                    clust = 10
                    time_step = X_train.shape[1]
                  
                    tf.reset_default_graph() 
                                 
                    X = tf.placeholder(tf.float32, [None, shp, 1])
                    Y_class = tf.placeholder(tf.float32, [None, 2])
                    Y_add = tf.placeholder(tf.float32, [None, 2])
                    Z = tf.placeholder(tf.float32, [None, noise])
                                    
                    ## discriminator 
                    p_real, p_logit_real, real_net  = proposed_bfgan.discriminator(X, Y_class, shp = shp,
                                                                         clust = clust,
                                                                         batch_size = batch_size,
                                                                         reuse = False)
                
                
                    fake = proposed_bfgan.generator(Z, Y_class, Y_add, shp = shp)
                    
                    p_fake, p_logit_fake, fake_net  = proposed_bfgan.discriminator(fake, Y_class, 
                                                                         shp = shp,
                                                                         clust = clust,
                                                                         batch_size = batch_size,
                                                                         reuse = True)
                        
                    ## Categorical GAN loss
                    d_loss = -proposed_bfgan.entropy1(p_real, batch_size)+proposed_bfgan.entropy2(p_real,
                                                  batch_size)-proposed_bfgan.entropy2(p_fake, batch_size)
                    g_loss = -proposed_bfgan.entropy1(p_fake, batch_size)+proposed_bfgan.entropy2(p_fake, batch_size)
                    
                    
                    ## Additional INformation loss
                                     
                    real_label = Y_add
                    
                    class_fake, class_logit_fake = proposed_bfgan.classifier(fake_net, classes = classes, reuse=False)
                    class_real, class_logit_real = proposed_bfgan.classifier(real_net, classes = classes, reuse=True)
                    
                    q_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = class_logit_real,
                                                                                         labels = real_label))
                    q_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = class_logit_fake,
                                                                                         labels = real_label))
                    
                    q_loss = q_real_loss + q_fake_loss                    
                    
                    '''
                    q_loss = tf.sigmoid(q_loss)
                    d_loss = tf.sigmoid(d_loss)
                    g_loss = tf.sigmoid(g_loss)
                    '''
                    
                    t_vars = tf.trainable_variables()
                    d_vars = [var for var in t_vars if 'discriminator' in var.name]
                    g_vars = [var for var in t_vars if 'generator' in var.name]
                    q_vars = [var for var in t_vars if ('discriminator' in var.name) or ('classifier' in var.name) or ('generator' in var.name)]
                    
                    train_D = tf.train.AdamOptimizer(0.0001, beta1 = 0.5).minimize(d_loss,
                                                    var_list = d_vars)
                    train_G = tf.train.AdamOptimizer(0.001, beta1 = 0.5).minimize(g_loss,
                                                    var_list = g_vars)     
                    train_Q = tf.train.AdamOptimizer(0.0001, beta1 = 0.5).minimize(q_loss,
                                                    var_list = q_vars)     
                    
                    ##############
                    total_batch = int(X_train.shape[0]/batch_size)*2
                            
                    init = tf.global_variables_initializer()
                    sess = tf.Session()
                    sess.run(init)
                    losses = {"d":[], "g":[], "q":[]}
                    
                    start = time()
                    print("start training GAN model")
                    print("total epoch :", total_epoch)
                
             
                    for epoch in range(total_epoch):
                        train_L_D = 0.
                        train_L_G = 0.
                        train_L_Q = 0.
                        
                        for i in range(total_batch):
                            
                            if i % 2 == 1:
                                maj_idx = np.random.randint(0, X_train[y_train[:,0]==1,:].shape[0], size = int(batch_size/2))
                                min_idx = np.random.randint(0, X_train[y_train[:,0]==0,:].shape[0], size = int(batch_size/2))
                                batch_x = np.concatenate((X_train[y_train[:,0]==1,:][maj_idx], X_train[y_train[:,0]==0,:][min_idx]), axis = 0)
                                batch_y = np.concatenate((y_train[y_train[:,0]==1,:][maj_idx], y_train[y_train[:,0]==0,:][min_idx]), axis = 0)
                            
                            if i % 2 == 0:
                                maj_idx = np.random.randint(0, X_train[y_train[:,2]==1,:].shape[0], size = int(batch_size/2))
                                min_idx = np.random.randint(0, X_train[y_train[:,2]==0,:].shape[0], size = int(batch_size/2))
                                batch_x = np.concatenate((X_train[y_train[:,2]==1,:][maj_idx], X_train[y_train[:,2]==0,:][min_idx]), axis = 0)
                                batch_y = np.concatenate((y_train[y_train[:,2]==1,:][maj_idx], y_train[y_train[:,2]==0,:][min_idx]), axis = 0)
                                
                            batch_y_class = batch_y[:,0:2]
                            batch_y_add = batch_y[:,2:4]

                            batch_noise = np.random.normal(0., 1., (batch_size, noise))
                                            
                            _, batch_L_D = sess.run([train_D, d_loss],
                                                    {X: batch_x,
                                                     Y_class: batch_y_class,
                                                     Y_add: batch_y_add,
                                                     Z: batch_noise})
                            
                            ## update D & G loss 
                            _, batch_L_G, _, batch_L_Q = sess.run([train_G, g_loss, train_Q, q_loss],
                                                     {Z: batch_noise,
                                                      Y_class: batch_y_class,
                                                      X: batch_x,
                                                      Y_add: batch_y_add})
                
                            train_L_D += batch_L_D
                            train_L_G += batch_L_G
                            train_L_Q += batch_L_Q
                            
                        train_L_D /= total_batch
                        train_L_G /= total_batch
                        train_L_Q /= total_batch
                        
                        losses["d"].append(train_L_D)
                        losses["g"].append(train_L_G)
                        losses["q"].append(train_L_Q)

                        '''                        
                        #if epoch % 10 == 0:
                            print("Epochs :","%0.0f" %epoch,"| D Loss :","%0.4f" %train_L_D, "| G Loss :", "%0.4f" %train_L_G, "| C Loss :", "%0.4f" %train_L_Q)
                            #sample_plot(plot_names="None", csv_names="None", save =False)
                        '''
                        print("Epochs :","%0.0f" %epoch,"| D Loss :","%0.4f" %train_L_D, "| G Loss :", "%0.4f" %train_L_G, "| C Loss :", "%0.4f" %train_L_Q)
                        
                    stop = time()
                    print("finished GAN model")
                    
                    plot_names = spath + "/plot/" + train_file[files-1][len(path)] + "_neighbor_{}_weights_{}_margins_{}".format(nn,w,m) +".png"
                    csv_names = spath + "/csv/" + train_file[files-1][len(path)] + "_neighbor_{}_weights_{}_margins_{}".format(nn,w,m)  + ".csv"
                    loss_names = spath + "/loss/" + train_file[files-1][len(path)] + "_neighbor_{}_weights_{}_margins_{}".format(nn,w,m)  + ".png"
    
                    proposed_bfgan.loss_plot(losses, names = loss_names, save = True)
    
                    print()
                    sample_plot(plot_names, csv_names, save = True)
    
                    print("save.. fin")
                    print("#################################################################")
                          
                else:
                    
                    print("skip_neighbor_{}_weights_{}_margins_{}".format(nn,w,m))
                    print("#################################################################")
                          
                          

            
            
            
            
            
            
                      
                               
