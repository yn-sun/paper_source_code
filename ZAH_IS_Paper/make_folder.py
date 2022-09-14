import os
import shutil
if __name__=='__main__':
    folder='log_rbos'
    for name in os.listdir(r'./keel/'):
        path='./keel/'+name+'/'+folder
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)
            print(path)
