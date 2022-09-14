# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 18:41:10 2021
@author: Zhan ao Huang
@email: huangzhan_ao@outlook.com
"""
import os
import shutil
if __name__=='__main__':
    for file in os.listdir(r'./keel/'):
        log_file='./keel/'+file+'/log_smote/log_smote.txt'
        print(log_file)
        # shutil.rmtree(log_file)
        if os.path.exists(log_file):
            os.remove(log_file)
            # shutil.rmtree(log_file)
            print('remove '+ log_file)
        else:
            print('no log file exists.')