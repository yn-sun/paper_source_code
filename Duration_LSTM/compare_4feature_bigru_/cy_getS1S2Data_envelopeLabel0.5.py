#-*- coding:utf8 -*-

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
from cyEcgLib import SearchS1S2MultiSample
#segment feature label to sample

########################################################################
def getS1S2DataList(NameList,EnvelopDict,LabelDict,HeartRateDict,sampleLength=200, sampleNumber=200):
    SAMPLE_DIM = 4
    #initialnize
    fs = 50
    heartSoundList=[]
    heartSoundLabelList =[]
    s1s2List=[]
    #recurrent for all file
    for fileName_pre in NameList:
        envFeature = EnvelopDict[fileName_pre].T
        envLabel = LabelDict[fileName_pre]
        HeartRateDictScaled = cyEcgLib.HeartRateDict_scale(HeartRateDict)
        #Add feature
        addFeature=np.zeros((envFeature.shape[0],2))
        for i in range(envFeature.shape[0]):
            addFeature[i][0]=HeartRateDictScaled[fileName_pre][0]#heart rate
            addFeature[i][1]=HeartRateDictScaled[fileName_pre][1]#heart duration
        print "envFeature", envFeature.shape,addFeature.shape
        #envFeature=np.hstack((envFeature,addFeature))

        print "envFeature.shape,envLabel.shape",envFeature.shape,envLabel.shape
        ms = SearchS1S2MultiSample()
        setFeature,segLabel=ms.resampFromLongFeature(envFeature, envLabel, sampleLength=sampleLength, sampleNumber=sampleNumber)
        #connect Two List
        heartSoundList=heartSoundList+setFeature
        heartSoundLabelList = heartSoundLabelList + segLabel
        print len(heartSoundList), len(heartSoundLabelList)
        print "setFeature.shape", len(heartSoundList)
    s1s2List=np.asarray(s1s2List).reshape(1,-1)
    print s1s2List
    heartSoundList=np.reshape(heartSoundList,(-1,sampleLength,SAMPLE_DIM))
    heartSoundLabelList=np.reshape(heartSoundLabelList,(-1,sampleLength,1))
    return heartSoundList,heartSoundLabelList,s1s2List

if __name__ == '__main__':
    EnvelopDict, HeartRateDict, Duration_distributionsDict, Duration_distributionsMaxMinDict = cyEcgLib.readEnvelopeMatToDict(
        Path='../../AMatdata/wavEnvelope50hz.mat')

    LabelDict = cyEcgLib.readLabelToDict(Path='../../AMatdata/wavLabel50hz.mat')
    removeList = ['376', '330', '309', '294', '269', '264', '253', '223', '204', '168', '155', '151', '413', '111',
                  '108', '099', '086', '074', '069', '066', '031']

    ms=SearchS1S2MultiSample()
    List=ms.getTheTrainSetAndTestSetList_DiscartError(LabelDict,removeList)
    trainNameList,testNameList=ms.splitTrainTest(List,0.5)

    TrainList,TrainLabelList,Trains1s2List=getS1S2DataList(trainNameList,EnvelopDict,LabelDict,HeartRateDict,200,200)
    TestList,TestLabelList,Tests1s2List=getS1S2DataList(testNameList,EnvelopDict,LabelDict,HeartRateDict,200,200)
    print TrainList.shape,TrainLabelList.shape
    print TestList.shape,TestLabelList.shape
    # # # 创建HDF5文件
    # # 写入
    ms.saveH5(fileName='./TrainTestData.h5',TrainList=TrainList, TrainLabelList=TrainLabelList, TestList=TestList, TestLabelList=TestLabelList)