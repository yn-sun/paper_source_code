import xlwt
import os
import numpy as np
def compute_metric(file,name):
    f=open(file)
    line=f.readline()
    count,fold=0,0
    max_auc=np.zeros(5)
    
    while line:
        if '[0,5000]' in line:
            while line:
                if '5000]' in line:
                    idx0=line.index(name)
                    test_auc=line[idx0+len(name):idx0+len(name)+5]
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
    print(name,max_auc,np.mean(max_auc))
    return np.mean(max_auc)

def compute_hidc_metric(file,name):
    if os.path.exists(file):
        f=open(file)
        count,fold,gamma=0,0,-5
        max_auc=np.zeros(5)
        line=f.readline()
        while line:
            if 'set gamma' in line:
                count=50
                while count:
                    if '5000]' in line:
                        idx0=line.index(name)
                        test_auc=line[idx0+len(name):idx0+len(name)+5]           
                        if float(test_auc)>max_auc[fold]:
                            max_auc[fold]=float(test_auc)
                    line=f.readline()
                    count-=1
    
                gamma+=1
                if gamma>4:
                    #print('----')
                    fold+=1
                    gamma=-5
                    
            line=f.readline()
        f.close()
        print('auc:',max_auc,np.mean(max_auc))
        return np.mean(max_auc)

def result_all_dataset(mtc):
    metrics={'auc':'test_auc=','f1_score':'test_f1_score=','g_mean':'test_g_mean=','macc':'test_macc='}
    methods=['ros','rus','smote','rbus','rbos','crius','nbus','ibp','bwl','hidc']
    #methods=['hidc']
    workbook=xlwt.Workbook(encoding='utf-8')
    worksheet=workbook.add_sheet('sheet1')
    
    name_rows=2
    method_cols=1
    #表头
    worksheet.write_merge(0,0,0,len(methods),label=mtc)
    for mtd in methods:
        if mtd=='hidc_ibp':
            worksheet.write(1,method_cols,label='ibp')    
        else:
            worksheet.write(1,method_cols,label=mtd)
        method_cols+=1
    
    for name in os.listdir(r'./keel/'):
        worksheet.write(name_rows,0,label=name)
        method_cols=1
        for mtd in methods:
            auc_file='./keel/'+name+'/log_'+mtd+'/log_'+mtd+'.txt'
            if mtd=='hidc':
                auc=compute_hidc_metric(auc_file,metrics[mtc])
                worksheet.write(name_rows,method_cols,label=auc)
            else:
                auc=compute_metric(auc_file,metrics[mtc])
                worksheet.write(name_rows,method_cols,label=auc)
            method_cols+=1
        name_rows+=1
    workbook.save(mtc+'.xls')
if __name__=='__main__':
    result_all_dataset('auc')
    result_all_dataset('f1_score')
    result_all_dataset('g_mean')
    result_all_dataset('macc')