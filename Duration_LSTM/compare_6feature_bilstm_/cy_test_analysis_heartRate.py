
from __future__ import print_function
import os
import sys
sys.path.append("../..")
if os.name =='nt':
    os.environ['THEANO_FLAGS'] ='floatX=float32,mode=FAST_RUN'
elif os.name == 'posix':
    os.environ['THEANO_FLAGS'] = "device=gpu0,floatX=float32,mode=FAST_RUN,cuda.root= '/usr/local/cuda-8.0/'"
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization,  Input,Embedding,wrappers,Permute,Reshape,Convolution2D
from keras.layers import LSTM, SimpleRNN, GRU,MaxPooling2D,TimeDistributed,Bidirectional
from keras.models import Model
import numpy as np
from cyPcgLib import DLSTM
import cyEcgLib
from matplotlib import pyplot

EnvelopDict, HeartRateDict, Duration_distributionsDict, Duration_distributionsMaxMinDict = cyEcgLib.readEnvelopeMatToDict(
    Path='../../AMatdata/wavEnvelope50hz.mat')
def HeartRateDict_scale(dict):
    outputDict={}
    list=[]
    for str in dict:
        list.append(dict[str].flatten())#1*1*2,flatten,1*2

    array=np.asarray(list).T
    #print array.shape
    mean_heartReat=np.mean(array[0])
    std_heartReat=np.std(array[0])

    mean_sistole_mean=np.mean(array[1])
    std_sistole=np.std(array[1])

    #print("mean_heart mean,std",mean_heartReat,std_heartReat)
    #print("mean_heart mean,std", mean_sistole_mean, std_sistole)
    for str in dict:
        outputDict[str]=[float(dict[str].flatten()[0]-mean_heartReat)/std_heartReat,float(dict[str].flatten()[1]-mean_sistole_mean)/std_sistole]
    return outputDict,array
print(HeartRateDict)
outputDict,array=HeartRateDict_scale(HeartRateDict)
array=np.asarray(array)
print(array.shape)

pyplot.hist(array[0],100)
pyplot.figure()
pyplot.hist(array[1],100)
pyplot.figure()
pyplot.scatter(array[0],array[1],norm=True)
pyplot.figure()
pyplot.boxplot(array[0])
pyplot.show()