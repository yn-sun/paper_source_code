import os
import shutil
if __name__=='__main__':
    cp_file_name='train_rbus.py'
    for name in os.listdir(r'./keel/'):
        target_file_name='abalone_3_vs_11_5_fold'
        if name !=target_file_name:
            file_name='./keel/'+name+'/'+cp_file_name
            f0=open('./keel/'+target_file_name+'/'+cp_file_name)
            f1=open('./'+cp_file_name,'w')
            line=f0.readline()
            while line:
                if "path='../" in line:
                    f1.write("    path='../"+name+"/'")
                    f1.write('\n')
                else:
                    f1.write(line)
                line=f0.readline()
            f0.close()
            f1.close()
            print('copy '+cp_file_name+' file to '+name)
            shutil.copyfile('./'+cp_file_name,'./keel/'+name+'/'+cp_file_name)
            
            
