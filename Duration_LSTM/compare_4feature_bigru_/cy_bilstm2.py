'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

'''
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
import cPickle
from keras.optimizers import SGD, Adam, RMSprop
import h5py
from sklearn.metrics import precision_score,roc_curve,f1_score,recall_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import logging

# input image dimensions


SaveName = "bilstm0.1"
optimizer = 'Adam'
TrainLoadWeights=False
TrainLoadWeightsName = SaveName #"bilstm0.1"

def build_BiLstm_softmax(inputTickdim,inputFeatureDim,outputTick, outputClass,loss='ctc_cost_for_train', optimizer='Adadelta', TrainLoadWeights = False,TrainLoadWeightsName = "bilstm0.1"):
    """
    Input shape: X.shape=(B, 1, rows, cols), GT.shape=(B, L)
    :param feadim: input feature dimension
    :param Nclass: class number
    :param loss:
    :param optimizer:
    :return:
    """
    net_input = Input(shape=(inputTickdim,inputFeatureDim))
    blstm0  = Bidirectional(LSTM(64, return_sequences=True, name='lstm0'))(net_input)
    blstm1  = Bidirectional(LSTM(64, return_sequences=True,  name='lstm1'))(blstm0)
    dense0 = TimeDistributed(Dense(outputClass, activation='softmax', name='dense0'))(blstm1)
    model  = Model(net_input, dense0)
    # load weights For Train
    if TrainLoadWeights == True:
        print("Load weight")
        model.load_weights(TrainLoadWeightsName + '.h5')
    #model.compile(loss=loss, optimizer=optimizer, sample_weight_mode='temporal')
    model.compile(loss=loss, optimizer=optimizer,metrics=['categorical_accuracy'])
    return model


if __name__ == '__main__':
    inputSecond = 2  # seconds
    batch_size = 32 * 2
    inputTickdim = inputSecond * 50
    inputFeatureDim = 4
    outputTick = inputSecond * 50
    outputClass = 4
    nb_epoch = 10

    Dlstm=DLSTM()
    secondLength=inputSecond
    sampleNumber=20
    generateList=False

    args = Dlstm.get_arguments(nb_epoch=nb_epoch,batch_size=batch_size,inputTickdim=inputTickdim,inputFeatureDim=inputFeatureDim,outputTick=outputClass,outputClass=outputClass,SaveName=SaveName,TrainLoadWeights=TrainLoadWeights,TrainLoadWeightsName=TrainLoadWeightsName)

    EnvelopDict, HeartRateDict, Duration_distributionsDict, Duration_distributionsMaxMinDict = cyEcgLib.readEnvelopeMatToDict(
        Path='../../AMatdata/wavEnvelope50hz.mat')

    LabelDict = cyEcgLib.readLabelToDict(Path='../../AMatdata/wavLabel50hz.mat')
    removeList = ['376', '330', '309', '294', '269', '264', '253', '223', '204', '168', '155', '151', '413', '111',
                  '108', '099', '086', '074', '069', '066', '031']

    ms = cyEcgLib.SearchS1S2MultiSample()
    List = ms.getTheTrainSetAndTestSetList_DiscartError(LabelDict, removeList)
    if generateList == False:
        trainNameList, testNameList = ms.splitTrainTest(List, 0.5)
        myFile = file(SaveName + '_my_train_test70.pkl', 'w')
        cPickle.dump([trainNameList, testNameList], myFile)
        myFile.close()
    else:
        trainNameList, testNameList = cPickle.load(open('my_train_test50.pkl'))

    # print(Y_train.shape)
    # print(Y_test[1:2])
    ##################end prepare data##########
    loss = 'categorical_crossentropy'
    optimizer = 'Adam'
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    print(outputClass)
    model = build_BiLstm_softmax(inputTickdim=args.input_length,inputFeatureDim=args.input_dim,outputTick=args.output_length, outputClass=args.output_dim,loss=loss, optimizer=optimizer,TrainLoadWeights = args.TrainLoadWeights,TrainLoadWeightsName = args.TrainLoadWeightsName)
    #returnResult =model.predict(X_train[1:4],batch_size=1)

    #print("returnResult",returnResult,returnResult.shape)
    #Train and save weight ,network artritecture##################################
    SaveName =args.SaveName
    TRAIN = args.Train
    if TRAIN == True:
        json_string = model.to_json()
        for i in range(args.epoch):
            print('Epoch', i, '/', args.epoch)
            print('Train...')
            X_train, X_valid, X_test, Y_train, Y_valid, Y_test = Dlstm.getData_FourFeature(trainNameList, testNameList,
                                                                                     EnvelopDict, LabelDict,HeartRateDict,
                                                                                     secondLength, sampleNumber,
                                                                                     inputTickdim, inputFeatureDim,
                                                                                     outputTick)
            history=model.fit(X_train, Y_train, batch_size=args.batch_size, epochs=1,
                      verbose=1,
                      shuffle=True,validation_data=(X_test, Y_test))
            allScore = []
            if (i > 0 ):
                #cel = cyChallengeLib.EcgEngryLib()
                #allScore = cel.getFileTestF1score_Modify(model, AllSig, AllLabel, testNameList, voteNum=64, cutLength=cutLength,segNum=3)
                allScore1=Dlstm.getF1score(model, X_test, Y_test, tolerate=5)
                allScore2=Dlstm.getF1score(model, X_test, Y_test, tolerate=3)
                # save epoch weights
                Dlstm.saveEpochLog(SaveName, inputSecond, allScore1,allScore2)
            Dlstm.saveEpochWeight(SaveName, model, i)
    ###############model load and compile
    else:
        print("load weights:")
        model = model_from_json(open(SaveName + '.json').read())
        model.load_weights(SaveName + '.h5')
        model.compile(loss=loss,
                      optimizer=optimizer, metrics=['accuracy'])
        ####################################################
    score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = Dlstm.getData_FourFeature(trainNameList, testNameList,
                                                                                     EnvelopDict, LabelDict,HeartRateDict,
                                                                                     secondLength, sampleNumber,
                                                                                     inputTickdim, inputFeatureDim,
                                                                                     outputTick)
    X_sample = X_test
    Y_sample = Y_test
    print("Model Predict Input Data")
    returnResult = model.predict(X_sample, batch_size=200)
    print(returnResult.shape)

    predictLabel = Dlstm.oneHotToNumber3D(returnResult)
    targetLabel = Dlstm.oneHotToNumber3D(Y_sample)
    directoryName = "Predict_s1s2position"
    if not os.path.exists(directoryName):
        os.makedirs(directoryName)
    print("save feature Data")
    np.save(directoryName + '/' + "feature.npy", np.asarray(X_sample))
    print("save predictLabel Data")
    np.save(directoryName + '/' + "predict.npy", np.asarray(predictLabel))
    print("save targetLabel Data")
    np.save(directoryName + '/' + "target.npy", np.asarray(targetLabel))


