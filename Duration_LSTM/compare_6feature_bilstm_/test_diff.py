import os
import matplotlib.pyplot as plt
import matplotlib.pyplot
import sys
import numpy as np
sys.path.append("../..")
import cyEcgLib
import cyMetricLib
import cyPcgLib
import seaborn as sns
sns.set_style("whitegrid")
fs=50
displaySecondNum=4

arrayXlabel=cyMetricLib.getArrayXticks(displaySecondNum,fs)#ax.set_xticks ' arrary,ex [0., fs*1, fs*2, fs*3,fs*4]
arrayXticksLabel=cyMetricLib.getArrayXticksLabel(displaySecondNum,fs)#ax set x label ,ex ax.set_xticklabels(["$0$", r"$\frac{1}{2}\pi$",
                     #r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])

directoryName="Predict_s1s2position"
predictAll=np.load(directoryName + '/'+"predict.npy")
targetAll=np.load(directoryName + '/'+"target.npy")
featureAll=np.load(directoryName + '/'+"feature.npy")
sampleNumber=990
predict=predictAll[sampleNumber]
a=np.diff(predict)
# for i in range(len(a)):
#     print a[i],predict[i]


def find_s2Position(y_predict):

    positionList=[]
    index = np.where(y_predict == 2)
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
print predict
positionList=find_s2Position(predict)
print positionList


def get_median_position(positionList):
    median_positionList=[]
    for i in positionList:
        median=i[0]+(i[1]-i[0])/2
        median_positionList.append(median)
    return median_positionList
print get_median_position(positionList)
