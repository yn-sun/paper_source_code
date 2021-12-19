import pickle

import h5py
#no filter
import random
import numpy as np
import sys
sys.path.append("../..")
import cyEcgLib
import cyMetricLib
import cyPcgLib
#segment feature label to sample
SAMPLE_LENGTH=200
SAMPLE_DIM=6
def resampFromLongFeature(Feature,Label,length=60,number=1000):
    segFeature=[]
    segLabel=[]
    start_index=np.random.randint(0,len(Feature)-length,(1,number))#generate n mumber start index for segmenting feature
    #print start_index
    for i in start_index[0]:
        segFeature.append(Feature[i:i+length])
        segLabel.append(Label[i:i+length])
    print len(segFeature),len(segLabel)
    return segFeature,segLabel
#get the train set and test set List from dict
def getTheTrainSetAndTestSetList(LabelDict):
    List=[]
    for i in LabelDict:
        List.append(i)
    return List
#some position is error ,we will discard these file
def getTheTrainSetAndTestSetList_DiscartError(LabelDict):
    List=getTheTrainSetAndTestSetList(LabelDict)
    removeList=['376','330','309','294','269','264','253','223','204','168','155','151','413','111','108','099','086','074','069','066','031']
    for i in removeList:
        removeItem='a0'+i

        try:
            List.remove(removeItem)
            print("delete item sucess:" + removeItem)
        except:
            print("delete error:"+removeItem)
        #print(len(List))
    return List
def splitTrainTest(List,train_size=0.5):
    from sklearn.utils import shuffle
    shuffle_List=shuffle(List)
    len_all=len(shuffle_List)
    split_number=int(len_all*train_size)

    trainSet=shuffle_List[0:split_number]
    testSet=shuffle_List[split_number:]
    return trainSet,testSet
LabelDict = cyEcgLib.readLabelToDict(Path='../../AMatdata/wavLabel50hz.mat')
List=getTheTrainSetAndTestSetList_DiscartError(LabelDict)
trainSet,testSet=splitTrainTest(List,0.5)
print(trainSet)
print(len(trainSet))
# def splitTrainTest_test(List):
#     from sklearn.model_selection import ShuffleSplit,StratifiedShuffleSplit
#     rs=ShuffleSplit(n_splits=3, train_size=0.5, test_size=.25,random_state=0)
#     for train_index, test_index in rs.split(List):
#         print("TRAIN:", train_index, "TEST:", test_index)
#         print(train_index.shape)
#     return train_index,test_index