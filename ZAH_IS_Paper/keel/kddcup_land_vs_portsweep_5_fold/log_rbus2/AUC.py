# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 10:43:57 2021
@author: Zhan ao Huang
@email: huangzhan_ao@outlook.com
"""
import numpy as np
if __name__=='__main__':
    f=open('./log_rbus2.txt')
    line=f.readline()
    count,fold=0,0
    max_auc=np.zeros(5)
    
    while line:
        if '[0,5000]' in line:
            while line:
                if '5000]' in line:
                    idx0=line.index('test_acu=')
                    test_auc=line[idx0+9:idx0+9+5                 ]
                    #print(test_auc)
                    if float(test_auc)>max_auc[fold]:
                        max_auc[fold]=float(test_auc)
                    if '[4900,5000]' in line:
                        fold+=1
                        if fold==5:
                            break 
                line=f.readline()
                
        line=f.readline()
    f.close()
    print('auc:',max_auc,np.mean(max_auc))