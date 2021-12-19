import numpy as np
y_predict=[0,0,0,1,1,1,2,2,3,3,0,0,1,1,2,2,2,2,3,3,1,1,2,2,2,2,3,3,1,1,2,2,2,2,3,3]
y_predict=np.asarray(y_predict)

index=np.where(y_predict==2)
print index
print np.diff(index)
print np.nonzero(np.diff(y_predict)==1)

def find_s2Position(y_predict,label=2):
    #predict list,get the start_position,and end position
    #input y_predict = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 1, 1, 2, 2, 2, 2, 3, 3, 1, 1, 2, 2, 2, 2,
    #             3, 3]
    #label=2
    #return:[[6, 7], [14, 17], [22, 25]]
    positionList=[]
    index = np.where(y_predict == label)
    index=index[0]
    min_bandary=index[0]
    for i in range(len(index)-1):#iterator all index array
        if (index[i+1]-index[i]==1):#the index is continious
            min_bandary=min_bandary
            max_bandary =index[i+1]
        else:                       #the index is continious,index[i+1]-index[i]!=1
            positionList.append([min_bandary,max_bandary])
            min_bandary = index[i + 1]
            max_bandary = index[i + 1]
    return positionList
print find_s2Position(y_predict)

