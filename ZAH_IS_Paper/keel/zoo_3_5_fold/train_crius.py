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
from sklearn.cluster import KMeans
import read_data as read_data

def generate(neg_data,size,attr_count):
    select_neg=np.zeros((size,attr_count))
    for i in range(size):
        idx=np.random.randint(0,len(neg_data))
        select_neg[i]=neg_data[idx]
    return select_neg

def ip(x,sigmma):
    n=len(x)
    rvalue=0
    for i in range(n):
        for j in range(n):
              rvalue+=math.e**(-((x[i]-x[j])**2).sum()/(4*sigmma**2))
    return rvalue/(n**2)
def cip(x,y,sigmma):
    rvalue=0
    for i in range(len(x)):
        for j in range(len(y)):
            rvalue+=math.e**(-((x[i]-y[j])**2).sum()/(4*sigmma**2))
    return rvalue/(len(x)*len(y))
def ris(x,y,sigmma,lam):#x \in y
    C=len(y)*cip(x,y,sigmma)/(len(x)*ip(x,sigmma))
    x_t=np.zeros(x.shape)
    for i in range(len(x)):
        f1,f2,f3,f4,f5,f6=0,0,0,0,0,0
        for j in range(len(y)):
            f2+=math.e**(-((x[i]-y[j])**2).sum()/(4*sigmma**2))
            f3+=(math.e**(-((x[i]-y[j])**2).sum()/(4*sigmma**2))*y[j])
        for k in range(len(x)):
            f1+=(math.e**(-((x[i]-x[k])**2).sum()/(4*sigmma**2))*x[k])
            f5+=math.e**(-((x[i]-x[k])**2).sum()/(4*sigmma**2))
        f4=f2
        f6=f2
        #print(f2,f4,f6,f5)
        x_t[i]=C*((1-lam)/lam)*f1/f2+f3/f4-C*((1-lam)/lam)*f5/f6*x[i]
    
    return x_t


def ris_undersampling(neg_data,attr_count,pos_count):
    select_neg=generate(neg_data,pos_count,attr_count)
    epoch=0
    while True:
        sigmma,lam=5,1000
        sigmma/=(epoch+1)
        select_neg2=ris(select_neg,neg_data,sigmma,lam)
        diff=((select_neg2-select_neg)**2).sum()
        print('ris undersampling: [%d,%.5f]'%(epoch,diff))
        select_neg=select_neg2
        if diff<=5e-5:
            break
        epoch+=1
    return select_neg
    
if __name__=='__main__':
    path='../zoo_3_5_fold/'
    train_data_all,test_data_all=read_data.read_data(path)
    cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count=read_data.construct_cv_trainloader(train_data_all,test_data_all)
    f=open(path+'log_crius/log_crius.txt','a')
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
        
        x0_array=np.array(x0).reshape(-1,attr_count)
        select_clusters_num=5
        clf=KMeans(n_clusters=len(x1)//select_clusters_num)
        clf.fit(x0_array)
        neg_pred_kmeans=clf.predict(x0_array)
        
        new_neg_data=[]
        for c in range(len(x1)//select_clusters_num):
            tmp_c=[]
            for i in range(len(neg_pred_kmeans)):
                if neg_pred_kmeans[i]==c:
                    tmp_c.append(x0_array[i])
            new_tmp_neg_data=ris_undersampling(np.array(tmp_c).reshape(-1,attr_count), attr_count, select_clusters_num)
            new_neg_data.append(new_tmp_neg_data)
        x0_new=np.array(new_neg_data).reshape(-1,attr_count)
        x1_array=np.array(x1).reshape(-1,attr_count)
        
        x=np.concatenate((x0_new,x1_array),axis=0)
        
        y0=np.zeros(len(x0_new))
        y1=np.ones(len(x1_array))
        y=np.concatenate((y0,y1),axis=0)
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
    