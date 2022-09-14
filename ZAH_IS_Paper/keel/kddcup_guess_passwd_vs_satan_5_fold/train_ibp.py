# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 14:27:23 2020

@author: Zhan ao Huang
"""

import numpy as np
import matplotlib.pyplot as plt
import Net as Net
import sys

import read_data as read_data
if __name__=='__main__':
    path='../kddcup_guess_passwd_vs_satan_5_fold/'
    train_data_all,test_data_all=read_data.read_data(path)
    cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count=read_data.construct_cv_trainloader(train_data_all,test_data_all)
    f=open(path+'log_ibp/log_ibp.txt','a')
    for fold in range(5):
        train_fold_data=cv_trainloader[fold]
        test_fold_data=cv_testloader[fold]
        x,y,x_test,y_test=[],[],[],[]
        
        for i,data in enumerate(train_fold_data):
            inputs,labels=data
            for j in range(len(labels)):
                x.append(list(inputs[j].numpy()))
                y.append(labels[j].item())
        x=np.array(x).reshape(-1,attr_count)
        y=np.array(y).reshape(-1,1)
        for i,data in enumerate(test_fold_data):
            inputs,labels=data
            for j in range(len(labels)):
                x_test.append(list(inputs[j].numpy()))
                y_test.append(labels[j].item())
        x_test=np.array(x_test).reshape(-1,attr_count)
        y_test=np.array(y_test).reshape(-1,1)
        
        EPOCH=5000
        print(str(fold+1)+' fold training')
        f.write(str(fold+1)+' fold training')
        f.write('\n')
        myNN=Net.NN(attr_count,4,2,0.01)
        for gamma in np.arange(1):
            print('set gamma=%.2f'%gamma)
            f.write(str(['set gamma=.%2f'%gamma]))
            f.write('\n')
            for epoch in range(EPOCH):
                myNN.expand_backpropagation(x,y,gamma)
                if epoch%100==0:
                    loss=myNN.loss(x,y)
                    auc=myNN.evaluate_auc(x_test,y_test)
                    f1=myNN.evaluate_f1_score(x_test,y_test)
                    acc,ppr,sen,spe,macc,g_mean=myNN.evaluate_confusion_matrix(x_test,y_test)
                    print('[%d,%d]'%(epoch,EPOCH),'loss=%.3f'%loss,
                        'test_auc=%.3f'%auc,
                        'test_f1_score=%.3f'%f1,                       
                        'test_macc=%.3f'%macc,
                        'test_g_mean=%.3f'%g_mean)
                    f.write(str(['[%d,%d]'%(epoch,EPOCH),'loss=%.3f'%loss,
                        'test_auc=%.3f'%auc,
                        'test_f1_score=%.3f'%f1,                       
                        'test_macc=%.3f'%macc,
                        'test_g_mean=%.3f'%g_mean]))
                    f.write('\n')
    f.close()
   