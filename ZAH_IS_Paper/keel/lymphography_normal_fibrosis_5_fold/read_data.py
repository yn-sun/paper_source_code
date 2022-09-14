# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:31:19 2021
@author: Zhan ao Huang
@email: huangzhan_ao@outlook.com
"""
import numpy as np
import os
import torch.utils.data as Data
import torch


def read_data(path):
    train_data=[[],[],[],[],[]]
    test_data=[[],[],[],[],[]]
    
    for name in os.listdir(path):
        if 'lymphography' in name:
            #train file
            fold=int(name[-8])
            #print(fold)
            if 'tra' in name:    
                f=open(path+name)
                line=f.readline()
                attr1=[]
                attr2=[]
                attr3=[]
                attr4=[]
                attr5=[]
                attr6=[]
                attr7=[]
                attr8=[]
                attr9=[]
                attr10=[]
                attr11=[]
                attr12=[]
                attr13=[]
                attr14=[]
                attr15=[]
                #print(len(attr1))
                while line:
                    if '@attribute Lymphatics' in line and len(attr1)==0:
                        attr1=line[23:-2].replace(' ','').split(',')
                    elif '@attribute Block_of_affere' in line and len(attr2)==0:
                        attr2=line[28:-2].replace(' ','').split(',')
                    elif '@attribute Bl_of_lymph_c' in line and len(attr3)==0:
                        attr3=line[26:-2].replace(' ','').split(',')
                    elif '@attribute Bl_of_lymph_s' in line and len(attr4)==0:
                        attr4=line[26:-2].replace(' ','').split(',')
                    elif '@attribute By_pass' in line and len(attr5)==0:
                        attr5=line[20:-2].replace(' ','').split(',')
                    elif '@attribute Extravasates' in line and len(attr6)==0:
                        attr6=line[25:-2].replace(' ','').split(',')
                    elif '@attribute Regeneration_of' in line and len(attr7)==0:
                        attr7=line[28:-2].replace(' ','').split(',')
                    elif '@attribute Early_uptake_in' in line and len(attr8)==0:
                        attr8=line[28:-2].replace(' ','').split(',')
                    elif '@attribute Changes_in_lym' in line and len(attr9)==0:
                        attr9=line[27:-2].replace(' ','').split(',')
                    elif '@attribute Defect_in_node' in line and len(attr10)==0:
                        attr10=line[27:-2].replace(' ','').split(',')
                    elif '@attribute Changes_in_node' in line and len(attr11)==0:
                        attr11=line[28:-2].replace(' ','').split(',')
                    elif '@attribute Changes_in_stru' in line and len(attr12)==0:
                        attr12=line[28:-2].replace(' ','').split(',')
                    elif '@attribute Special_forms' in line and len(attr13)==0:
                        attr13=line[26:-2].replace(' ','').split(',')
                    elif '@attribute Dislocation_of' in line and len(attr14)==0:
                        attr14=line[27:-2].replace(' ','').split(',')
                    elif '@attribute Exclusion_of_no' in line and len(attr15)==0:
                        attr15=line[28:-2].replace(' ','').split(',')
                    
                    elif '@data' in line:
                        line=f.readline()

                        while line:
                            #for attr_1 in line
                            count=0.
                            for attr in attr1:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                    #print(line)
                                count+=1
                            
                            count=0.
                            for attr in attr2:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                    #print(line)
                                count+=1
                            
                            count=0.
                            for attr in attr3:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            
                            count=0.
                            for attr in attr4:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            
                            count=0.
                            for attr in attr5:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                                
                            count=0.
                            for attr in attr6:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                                
                            count=0.
                            for attr in attr7:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            count=0.
                            for attr in attr8:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                                
                            count=0.
                            for attr in attr9:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            count=0.
                            for attr in attr10:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                                
                            count=0.
                            for attr in attr11:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                                
                            count=0.
                            for attr in attr12:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                                
                            count=0.
                            for attr in attr13:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                                
                            count=0.
                            for attr in attr14:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                                
                            count=0.
                            for attr in attr15:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            
                            count=0
                            for attr in ['negative\n','positive\n']:
                                #print(line.replace(' ','').split(','))
                                if attr in line.replace(' ','').split(','):
                                    
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                        
                            train_data[fold-1].append(line.strip('\n').split(','))
                            line=f.readline()
                    
                    line=f.readline()
                f.close()
            elif 'tst' in name:
                
                f=open(path+name)
                line=f.readline()
                attr1=[]
                attr2=[]
                attr3=[]
                attr4=[]
                attr5=[]
                attr6=[]
                attr7=[]
                attr8=[]
                attr9=[]
                attr10=[]
                attr11=[]
                attr12=[]
                attr13=[]
                attr14=[]
                attr15=[]
                #print(len(attr1))
                while line:
                    if '@attribute Lymphatics' in line and len(attr1)==0:
                        attr1=line[23:-2].replace(' ','').split(',')
                    elif '@attribute Block_of_affere' in line and len(attr2)==0:
                        attr2=line[28:-2].replace(' ','').split(',')
                    elif '@attribute Bl_of_lymph_c' in line and len(attr3)==0:
                        attr3=line[26:-2].replace(' ','').split(',')
                    elif '@attribute Bl_of_lymph_s' in line and len(attr4)==0:
                        attr4=line[26:-2].replace(' ','').split(',')
                    elif '@attribute By_pass' in line and len(attr5)==0:
                        attr5=line[20:-2].replace(' ','').split(',')
                    elif '@attribute Extravasates' in line and len(attr6)==0:
                        attr6=line[25:-2].replace(' ','').split(',')
                    elif '@attribute Regeneration_of' in line and len(attr7)==0:
                        attr7=line[28:-2].replace(' ','').split(',')
                    elif '@attribute Early_uptake_in' in line and len(attr8)==0:
                        attr8=line[28:-2].replace(' ','').split(',')
                    elif '@attribute Changes_in_lym' in line and len(attr9)==0:
                        attr9=line[27:-2].replace(' ','').split(',')
                    elif '@attribute Defect_in_node' in line and len(attr10)==0:
                        attr10=line[27:-2].replace(' ','').split(',')
                    elif '@attribute Changes_in_node' in line and len(attr11)==0:
                        attr11=line[28:-2].replace(' ','').split(',')
                    elif '@attribute Changes_in_stru' in line and len(attr12)==0:
                        attr12=line[28:-2].replace(' ','').split(',')
                    elif '@attribute Special_forms' in line and len(attr13)==0:
                        attr13=line[26:-2].replace(' ','').split(',')
                    elif '@attribute Dislocation_of' in line and len(attr14)==0:
                        attr14=line[27:-2].replace(' ','').split(',')
                    elif '@attribute Exclusion_of_no' in line and len(attr15)==0:
                        attr15=line[28:-2].replace(' ','').split(',')
                    
                    elif '@data' in line:
                        line=f.readline()

                        while line:
                            #for attr_1 in line
                            count=0.
                            for attr in attr1:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                    #print(line)
                                count+=1
                            
                            count=0.
                            for attr in attr2:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                    #print(line)
                                count+=1
                            
                            count=0.
                            for attr in attr3:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            
                            count=0.
                            for attr in attr4:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            
                            count=0.
                            for attr in attr5:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            count=0.
                            for attr in attr6:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            count=0.
                            for attr in attr7:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            count=0.
                            for attr in attr8:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            count=0.
                            for attr in attr9:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            count=0.
                            for attr in attr10:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            count=0.
                            for attr in attr11:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            count=0.
                            for attr in attr12:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            count=0.
                            for attr in attr13:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            count=0.
                            for attr in attr14:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            count=0.
                            for attr in attr15:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            
                            count=0
                            for attr in ['negative\n','positive\n']:
                                if attr in line.replace(' ','').split(','):
                                    lst_line=line.replace(' ','').split(',')
                                    idx=lst_line.index(attr)
                                    lst_line[idx]=str(count)
                                    line=','.join(lst_line)
                                count+=1
                            
                            test_data[fold-1].append(line.strip('\n').split(','))
                            line=f.readline()
                    
                    line=f.readline()
                f.close()
                
    for i in range(5):
        for j in range(len(train_data[i])):
            for k in range(len(train_data[i][j])):
                
                train_data[i][j][k]=float(train_data[i][j][k])
                
    for i in range(5):
        for j in range(len(test_data[i])):
            for k in range(len(test_data[i][j])):
                test_data[i][j][k]=float(test_data[i][j][k])
    #print(train_data[1],test_data[1])
    return train_data,test_data
def construct_cv_trainloader(train_data_all,test_data_all):
    cv_trainloader=[]
    cv_testloader=[]
    cv_pos_count,cv_neg_count=[],[]
    for i in range(5):
        train_data,train_label=[],[]
        test_data,test_label=[],[]
        pos_count,neg_count=0,0
        for j in range(len(train_data_all[i])):
            train_label.append(train_data_all[i][j][-1])
            train_data.append(train_data_all[i][j][0:-1])
            if train_data_all[i][j][-1]==1:
                pos_count+=1
            else:
                neg_count+=1
        for k in range(len(test_data_all[i])):
            test_data.append(test_data_all[i][k][0:-1])
            test_label.append(test_data_all[i][k][-1])
        cv_pos_count.append(pos_count)
        cv_neg_count.append(neg_count)
        #trainloader
        tensor_train_data=torch.from_numpy(np.array(train_data).reshape(-1,len(train_data[0]))).float()
        mean=torch.mean(tensor_train_data)
        std=torch.std(tensor_train_data)
        tensor_train_data=(tensor_train_data-mean)/std
        tensor_train_label=torch.from_numpy(np.array(train_label).reshape(-1,1))
        torch_train_dataset=Data.TensorDataset(tensor_train_data,tensor_train_label)
        trainloader=Data.DataLoader(dataset=torch_train_dataset,
                                    batch_size=1,
                                    shuffle=True)
        #testloader
        tensor_test_data=torch.from_numpy(np.array(test_data).reshape(-1,len(test_data[0]))).float()
        tensor_test_data=(tensor_test_data-mean)/std
        tensor_test_label=torch.from_numpy(np.array(test_label).reshape(-1,1))
        torch_test_dataset=Data.TensorDataset(tensor_test_data,tensor_test_label)
        testloader=Data.DataLoader(dataset=torch_test_dataset,
                                    batch_size=1,
                                    shuffle=False)
        cv_trainloader.append(trainloader)
        cv_testloader.append(testloader)
    attr_count=len(test_data[0])
    #print(attr_count)
    print('pos count:',cv_pos_count,',neg count:',cv_neg_count)
    return cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count
if __name__=='__main__':
    train_data_all,test_data_all=read_data()
    #print(train_data_all[0])
    cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count=construct_cv_trainloader(train_data_all,test_data_all)
    #print(cv_trainloader,cv_testloader)