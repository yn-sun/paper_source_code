'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

'''
from __future__ import print_function
import os
#if os.name =='nt':
#    os.environ['THEANO_FLAGS'] ='floatX=float32,mode=FAST_RUN'
#elif os.name == 'posix':
 #   os.environ['THEANO_FLAGS'] = "device=gpu0,floatX=float32,mode=FAST_RUN,cuda.root= '/usr/local/cuda-8.0/'"
import sys
sys.path.append("../..")
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization,  Input,Embedding,wrappers,Permute,Reshape,Convolution2D
from keras.layers import LSTM, SimpleRNN, GRU,MaxPooling2D,TimeDistributed,Bidirectional,Convolution1D,AtrousConvolution1D,MaxPooling1D,Conv1D
from cyPcgLib import CLSTM,DLSTM
from keras.models import Model
import numpy as np
import cyEcgLib
import cPickle
import cyMetricLib

# input image dimensions




def one_test(inputSecond=2):
    Dlstm = DLSTM()
    inputSecond = inputSecond
    SaveName = "bilstm"+str(inputSecond)
    optimizer = 'Adam'
    TrainLoadWeights = False
    TrainLoadWeightsName = SaveName
    dictory = SaveName + '/'
    jsonName = dictory + SaveName

    sampleNumber=20
    batch_size = 32
    inputTickdim = 50 *inputSecond
    inputFeatureDim =4
    outputTick = 50*inputSecond
    outputClass = 4
    nb_epoch = 30

    f_score_oneTest=[]
    loss_oneTest=[]

    EnvelopDict, HeartRateDict, Duration_distributionsDict, Duration_distributionsMaxMinDict = cyEcgLib.readEnvelopeMatToDict(
        Path='../../AMatdata/wavEnvelope50hz.mat')

    LabelDict = cyEcgLib.readLabelToDict(Path='../../AMatdata/wavLabel50hz.mat')
    args = Dlstm.get_arguments(nb_epoch=nb_epoch,batch_size=batch_size,inputTickdim=inputTickdim,inputFeatureDim=inputFeatureDim,outputTick=outputClass,outputClass=outputClass,SaveName=SaveName,TrainLoadWeights=TrainLoadWeights,TrainLoadWeightsName=TrainLoadWeightsName)
    trainNameList, testNameList = cPickle.load(open("my_train_test70.pkl"))

    ##################end prepare data##########
    loss = 'categorical_crossentropy'
    optimizer = 'Adam'
    print(outputClass)
    #Train and save weight ,network artritecture##################################
    SaveName =args.SaveName
    TRAIN = args.Train
    print(len(trainNameList), len(testNameList))

    print("load weights:")
    model = model_from_json(open(jsonName + '.json').read())
    for i in range(0,29):
        weightName = dictory + "-"+str(i)+"-"
        model.load_weights(weightName + '.h5')
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test = Dlstm.getData_FourFeature(trainNameList, testNameList,
                                                                                       EnvelopDict, LabelDict,
                                                                                       HeartRateDict,
                                                                                       inputSecond, sampleNumber,
                                                                                       inputTickdim, inputFeatureDim,
                                                                                       outputTick)
        allScore1 = Dlstm.getF1score(model, X_test, Y_test, tolerate=5, convolution=False)
        allScore2 = Dlstm.getF1score(model, X_test, Y_test, tolerate=3, convolution=False)
        allScore3 = Dlstm.getF1score(model, X_test, Y_test, tolerate=2, convolution=False)
        # save epoch weights
        Dlstm.saveEpochLog(SaveName, inputSecond, allScore1, allScore2)
        f_score_oneTest.append([allScore1, allScore2, allScore3])
    return f_score_oneTest, loss_oneTest




if __name__ == '__main__':
    all_score=[]
    all_loss=[]
    for i in range(2,9):
        f_score_oneTest,loss_oneTest=one_test(inputSecond=i)
        all_score.append(f_score_oneTest)
        all_loss.append(loss_oneTest)
        myFile = file('Train_Record_ReGet.pkl', 'w')
        cPickle.dump([all_score, all_loss], myFile)
        myFile.close