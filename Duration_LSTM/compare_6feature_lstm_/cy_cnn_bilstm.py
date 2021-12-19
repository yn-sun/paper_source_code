'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

'''
from __future__ import print_function
import os
if os.name =='nt':
    os.environ['THEANO_FLAGS'] ='floatX=float32,mode=FAST_RUN'
elif os.name == 'posix':
    os.environ['THEANO_FLAGS'] = "device=gpu3,floatX=float32,mode=FAST_RUN,cuda.root= '/usr/local/cuda-8.0/'"
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization,  Input,Embedding,wrappers,Permute,Reshape,Convolution2D
from keras.layers import LSTM, SimpleRNN, GRU,MaxPooling2D,TimeDistributed,Bidirectional
from keras.models import Model
import numpy as np
import argparse
from keras.optimizers import SGD, Adam, RMSprop
import h5py
from sklearn.metrics import precision_score,roc_curve,f1_score,recall_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
batch_size = 32*2
inputTickdim=200
inputFeatureDim=6
outputTick =200
outputClass=4
nb_epoch = 4
# input image dimensions

SaveName = "cnn_bilstm0.1"
optimizer = 'Adam'
TrainLoadWeights=False
TrainLoadWeightsName = SaveName #"bilstm0.1"

def build_BiLstm_softmax_2(inputTickdim,inputFeatureDim,outputTick, outputClass,loss='ctc_cost_for_train', optimizer='Adadelta', TrainLoadWeights = False,TrainLoadWeightsName = "bilstm0.1"):
    """
    Input shape: X.shape=(B, 1, rows, cols), GT.shape=(B, L)
    :param feadim: input feature dimension
    :param Nclass: class number
    :param loss:
    :param optimizer:
    :return:
    """
    net_input = Input(shape=(1,inputTickdim,inputFeatureDim))
    reshape_input=Reshape((inputTickdim,inputFeatureDim,1))(net_input)# For Tensorflow
    cnn=Convolution2D(1,2,2,border_mode='same',activation='relu')(reshape_input)
    flatten_cnn=Flatten()(cnn)
    reshape_cnn=Reshape((200,6))(flatten_cnn)
    blstm0  = Bidirectional(LSTM(64, return_sequences=True, name='lstm0'))(reshape_cnn)
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
def build_BiLstm_softmax(inputTickdim,inputFeatureDim,outputTick, outputClass,loss='ctc_cost_for_train', optimizer='Adadelta', TrainLoadWeights = False,TrainLoadWeightsName = "bilstm0.1"):
    """
    Input shape: X.shape=(B, 1, rows, cols), GT.shape=(B, L)
    :param feadim: input feature dimension
    :param Nclass: class number
    :param loss:
    :param optimizer:
    :return:
    """
    net_input = Input(shape=(1,inputTickdim,inputFeatureDim))
    reshape_input=Reshape((inputTickdim,inputFeatureDim,1))(net_input)# For Tensorflow
    cnn=Convolution2D(16,4,4,border_mode='same',activation='relu')(reshape_input)
    flatten_cnn=Flatten()(cnn)
    reshape_cnn=Reshape((200,96))(flatten_cnn)
    blstm0  = Bidirectional(LSTM(64, return_sequences=True, name='lstm0'))(reshape_cnn)
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
def to_categorical(y, Tick,nb_classes=None):

    '''Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.
    2D->3D
    # Arguments
        y: class vector to be converted into a matrix
        nb_classes: total number of classes

    # Returns
        A binary matrix representation of the input.
    '''
    if not nb_classes:
        nb_classes = np.max(y.shape[2])+1
    Y = np.zeros((len(y), Tick,nb_classes))

    for i in range(len(y)):
        for j in range(Tick):
            Y[i, j,y[i,j]] = 1.
    return Y
def ReadData():
    # Load data and shaffle data
    file=h5py.File('TrainTestData.h5','r')
    trainData=file['TrainList']
    trainLabel=file['TrainLabelList']
    testData=file['TestList']
    testLable=file['TestLabelList']
    numTrain = len(trainData)

    # with open('S1S2Dict.pickle','rb')as f:
    #     dataAll=pickle.load(f)
    #     (trainData, trainDataLabel, testData, testDataLabel)=dataAll
    numTrain = len(trainData)

    def PreProcessing(data,dataLabel):
        numData=len(data)
        print(np.asarray(data).shape)
        data=np.asarray(data).reshape((numData,inputTickdim*inputFeatureDim))
        lable=np.asarray(dataLabel,dtype=int).reshape((data.shape[0],outputTick))#reshape and translate into interger
        print(data.shape,lable.shape)
        return data,lable
    print("Load from hdf5 file,the shap is ")
    trainData,trainLabel=PreProcessing(trainData,trainLabel)
    testData,testLable=PreProcessing(testData,testLable)


    #shuffle the data
    trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
    #generate valide data
    numSplit=int(round(0.9*len(trainData)))
    X_train=trainData[:numSplit]
    y_train=trainLabel[:numSplit]
    X_valid=trainData[numSplit:numTrain]
    y_valid=trainLabel[numSplit:numTrain]

    X_test=testData
    y_test=testLable

    #reshape data
    #print(inputTickdim,inputFeatureDim,)
    #print(X_train.shape)
    X_train=np.reshape(X_train,(-1,1,inputTickdim,inputFeatureDim))
    X_valid=np.reshape(X_valid,(-1,1,inputTickdim,inputFeatureDim))
    X_test=np.reshape(X_test,(-1,1,inputTickdim,inputFeatureDim))

    Y_train = y_train.reshape((-1,outputTick,1))
    Y_valid= y_valid.reshape((-1,outputTick,1))
    Y_test =y_test.reshape((-1,outputTick,1))

    Y_train = to_categorical(y_train,outputTick, outputClass)
    Y_valid= to_categorical(y_valid, outputTick, outputClass)
    Y_test = to_categorical(y_test, outputTick, outputClass)
    print("The output shape is: ")
    print('X_train shape:', X_train.shape)
    print('y train target',y_train.shape )
    print('Y train target',Y_train.shape )
    return X_train,X_valid,X_test,Y_train,Y_valid,Y_test
def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='Pcg dectect S1 S2')
    parser.add_argument('--epoch', type=int, default=nb_epoch,
                        help='How many wav files to process at once.')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='How many wav files to process at once.')
    parser.add_argument('--input_length', type=int, default=inputTickdim,
                        help='How many wav files to process at once.')
    parser.add_argument('--input_dim', type=int, default=inputFeatureDim,
                        help='How many wav files to process at once.')
    parser.add_argument('--output_length', type=int, default=outputTick,
                        help='How many wav files to process at once.')
    parser.add_argument('--output_dim', type=int, default=outputClass,
                        help='How many wav files to process at once.')
    parser.add_argument('--SaveName', type=str, default=SaveName,
                        help='Save weight Name')
    parser.add_argument('--Train', type=_str_to_bool, default=True,
                        help='Train? if Train=False,the program will not train ,will load savePreNme.weight ,and evaluate the test ')

    parser.add_argument('--TrainLoadWeights', type=_str_to_bool, default=TrainLoadWeights,
                        help='Train? if Train=False,the program will not train ,will load savePreNme.weight ,and evaluate the test ')
    parser.add_argument('--TrainLoadWeightsName', type=str, default=TrainLoadWeightsName,
                        help='SavePreName')
    TrainLoadWeights
    TrainLoadWeightsName
    return parser.parse_args()
if __name__ == '__main__':
    args = get_arguments()

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test=ReadData()
    # print(Y_train.shape)
    # print(Y_test[1:2])
    ##################end prepare data##########
    loss = 'categorical_crossentropy'
    optimizer = 'Adam'
    print(outputClass)
    model = build_BiLstm_softmax(inputTickdim=args.input_length,inputFeatureDim=args.input_dim,outputTick=args.output_length, outputClass=args.output_dim,loss=loss, optimizer=optimizer,TrainLoadWeights = args.TrainLoadWeights,TrainLoadWeightsName = args.TrainLoadWeightsName)
    #returnResult =model.predict(X_train[1:4],batch_size=1)

    #print("returnResult",returnResult,returnResult.shape)
    #Train and save weight ,network artritecture##################################
    SaveName =args.SaveName
    TRAIN = args.Train
    if TRAIN==True:
        print('Train...')
        model.fit(X_train, Y_train, batch_size=args.batch_size, nb_epoch=args.epoch,validation_data=(X_valid, Y_valid))
        json_string = model.to_json()

        open(SaveName + '.json', 'w').write(json_string)
        model.save_weights(SaveName + '.h5')
    ###############model load and compile
    else:
        print("load weights:")
        model = model_from_json(open(SaveName + '.json').read())
        model.load_weights(SaveName + '.h5')
        model.compile(loss=loss,
                      optimizer=optimizer,metrics=['accuracy'])
    ##########################################
    score, acc = model.evaluate(X_test, Y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)



