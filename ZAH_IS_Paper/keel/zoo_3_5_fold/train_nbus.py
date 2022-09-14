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

def dis_compute(data0,data1):
    return np.sqrt(((data0-data1)**2).sum()) 

def nbus(neg_data,cv_neg_count,cv_pos_count):
    density=np.zeros(cv_neg_count)
    for i in range(len(neg_data)):
        for j in range(len(neg_data)):
            if j!=i:
                _dis=dis_compute(neg_data[i],neg_data[j])
                density[i]+=np.e**(-_dis**2)
    
    #compute distance for each instance
    distance=np.zeros(cv_neg_count)
    for i in range(len(neg_data)):
        if density[i]==max(density):
            tmp_dis=[]
            for j in range(len(neg_data)):
                if j!=i:
                    _dis=dis_compute(neg_data[i],neg_data[j])
                    tmp_dis.append(_dis)
            distance[i]=max(tmp_dis)
        else:
            tmp_dis=[]
            for j in range(len(neg_data)):
                if density[i]<density[j] and j!=i:
                    _dis=dis_compute(neg_data[i],neg_data[j])
                    tmp_dis.append(_dis)
            distance[i]=min(tmp_dis)
    factor=(density**2)*distance
    new_neg_data=np.zeros((cv_pos_count,1,attr_count))
    for k in range(len(new_neg_data)):
        max_idx=np.argmax(factor)
        new_neg_data[k]=neg_data[max_idx]
        factor[max_idx]=np.min(factor)
    return np.array(new_neg_data)
        
if __name__=='__main__':
    path='../zoo_3_5_fold/'
    train_data_all,test_data_all=read_data.read_data(path)
    cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count=read_data.construct_cv_trainloader(train_data_all,test_data_all)
    f=open(path+'log_nbus/log_nbus.txt','a')
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

        #print(y.shape)
        x0,x1=[],[]
        
        for i in range(len(x)):
            if y[i][0]==1:
                x1.append(x[i])
            else:
                x0.append(x[i])

        x0_new=nbus(x0,len(x0),len(x1)).reshape(-1,attr_count)
        x1_array=np.array(x1).reshape(-1,attr_count)
        
        x=np.concatenate((x0_new,x1_array),axis=0)
        y0=np.zeros(len(x0_new))
        y1=np.ones(len(x1))
        y=np.concatenate((y0,y1),axis=0)
        
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
    