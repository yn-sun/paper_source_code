# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 10:43:57 2021
@author: Zhan ao Huang
@email: huangzhan_ao@outlook.com
"""
import numpy as np
if __name__=='__main__':
    f=open('./log2.txt')
    
    count,fold,gamma=0,0,-5
    max_auc=np.zeros(5)
    line=f.readline()
    while line:
        if 'set gamma' in line:
            count=50
            if gamma==0:
                while count:
                    if '5000]' in line:
                        #print(line)
                        idx0=line.index('test_macc=')
                        test_auc=line[idx0+10:idx0+10+5]           
                        if float(test_auc)>max_auc[fold]:
                            max_auc[fold]=float(test_auc)
                    line=f.readline()
                    count-=1
            while count:
                line=f.readline()
                count-=1
            gamma+=1
            if gamma>4:
                #print('----')
                fold+=1
                gamma=-5
                
        line=f.readline()
    f.close()
    print('macc:',max_auc,np.mean(max_auc))