# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 14:27:23 2020

@author: Zhan ao Huang
"""

import numpy as np
import matplotlib.pyplot as plt
import Net as Net
import sys
import math
import read_data as read_data

def potential(x,M,m,gamma):
    rvalue=0
    for i in range(len(M)):
        
        rvalue+=math.e**(-((((M[i]-x)**2).sum())**0.5/gamma)**2)
    for i in range(len(m)):
        rvalue-=math.e**(-((((m[i]-x)**2).sum())**0.5/gamma)**2)
    return rvalue
def RBOS(M,m,gamma,alpha,iteration,p):
    S=[]
    
    while len(S)+len(m)<len(M):
        idx=np.random.randint(0,len(m))
        x=m[idx]
        
        print('synthetic count:[%d,%d]'%(len(S)+len(m),len(M)))
        for i in range(iteration):
            rp=np.random.uniform(0,1)
            if rp<p:
                break
           
            random_direction=np.random.random((len(x)))
            #print(random_direction,'direction')
            x_translated=x+alpha*random_direction
            if potential(x_translated,M,m,gamma)>potential(x,M,m,gamma):
                x=x_translated
        S.append(x)
    return np.array(S)
    
if __name__=='__main__':
    path='../kddcup_land_vs_satan_5_fold/'
    train_data_all,test_data_all=read_data.read_data(path)
    cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count=read_data.construct_cv_trainloader(train_data_all,test_data_all)
    f=open(path+'log_rbos/log_rbos.txt','a')
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
        
        
        x0,x1=[],[]
        
        for i in range(len(x)):
            if y[i][0]==1:
                x1.append(x[i])
            else:
                x0.append(x[i])
        
        x_p=RBOS(x0,x1,0.05,0.01,20,0.02)
        x1_all=np.concatenate((x_p,x1),axis=0)
        
        x=np.concatenate((x1_all,x0),axis=0)
        y0=np.zeros(len(x0))
        y1=np.ones(len(x1_all))
        y=np.concatenate((y1,y0),axis=0)   
        print(len(x),len(y))
        x=x.reshape(-1,attr_count)
        y=y.reshape(-1,1)
        
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
        for epoch in range(EPOCH):
            myNN.batch_backpropagation(x,y)
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
    