#-*- coding:utf8 -*-
from __future__ import division
import os
import wfdb
import csv
from features import logfbank

import h5py
from sklearn.utils import shuffle
from keras.utils import to_categorical
import random
import argparse
import scipy
import numpy
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input, Embedding, wrappers, Permute, \
    Reshape, Convolution2D, GlobalAveragePooling1D
from keras.layers import LSTM, SimpleRNN, GRU, MaxPooling2D, TimeDistributed, Bidirectional, Convolution1D, \
    AtrousConvolution1D, MaxPooling1D, Conv1D, Conv2D
class ChallengeEcgFun:
    rootdir = "../Challenge2017"
    fs = 300
    noise_exclude_c = ['A00196', 'A00629', 'A00660', 'A01529', 'A01755', 'A01888', 'A02196', 'A02411', 'A03117',
                       'A03934', 'A04257', 'A04366', 'A04673', 'A04685', 'A04946', 'A05134', 'A05233', 'A05341',
                       'A05766', 'A05992', 'A06262', 'A06534', 'A06622', 'A06862', 'A06874', 'A06897', 'A06982',
                       'A07001', 'A07382', 'A07625', 'A08133', 'A08204', 'A00849', 'A00106', 'A00196', 'A00307',
                       'A06629', 'A00660', 'A01048', 'A01755', 'A01888', 'A03117', 'A03745', 'A03934', 'A00485',
                       'A05341', 'A05766', 'A05992', 'A06262', 'A06534', 'A06622', 'A06862', 'A06897,7001', 'A07382',
                       'A07625', 'A07736', 'A07764', 'A08112', 'A08402', 'A00106']

    def get_arguments(self, nb_epoch, batch_size, SaveName, IsLoadWeights, LoadWeightsName):
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
        parser.add_argument('--SaveName', type=str, default=SaveName,
                            help='Save weight Name')
        parser.add_argument('--Train', type=_str_to_bool, default=True,
                            help='Train? if Train=False,the program will not train ,will load savePreNme.weight ,and evaluate the test ')
        parser.add_argument('--IsLoadWeights', type=_str_to_bool, default=IsLoadWeights,
                            help='Train? if Train=False,the program will not train ,will load savePreNme.weight ,and evaluate the test ')
        parser.add_argument('--LoadWeightsName', type=str, default=LoadWeightsName,
                            help='SavePreName')

        return parser.parse_args()

    def saveH5(self, fileName, TrainList, TrainLabelList, TestList, TestLabelList):
        file = h5py.File(fileName, 'w')
        print(TrainList.shape, TrainLabelList.shape, TestList.shape, TestLabelList.shape)
        file.create_dataset('TrainList', data=TrainList)
        file.create_dataset('TrainLabelList', data=TrainLabelList)
        file.create_dataset('TestList', data=TestList)
        file.create_dataset('TestLabelList', data=TestLabelList)
        # file.create_dataset('Trains1s2List', data=Trains1s2List)
        # file.create_dataset('Tests1s2List', data=Tests1s2List)
        file.close()

    def getAllSig(self, rootdir="../Challenge2017"):
        AllSig = {}
        for parent, dirnames, filenames in os.walk(rootdir):  # traverse the rootdir
            print(dirnames)
            for filename in filenames:

                if (filename[-3:] == 'hea'):
                    # print(filename)
                    file = os.path.join(parent, filename)
                    sig, fields = wfdb.srdsamp(file[:-4])
                    # print(sig.shape)
                    fileShortName = filename[:-4]
                    AllSig[fileShortName] = sig
        return AllSig

    def getAllMatSig(self, rootdir="../Challenge2017"):
        import scipy.io as sio
        AllSig = {}
        for parent, dirnames, filenames in os.walk(rootdir):  # traverse the rootdir
            # print dirnames
            for filename in filenames:
                if (filename[-3:] == 'mat'):
                    # print(filename)
                    file = os.path.join(parent, filename)
                    sig = sio.loadmat(file[:-4])
                    sig = sig['val']
                    sig = sig.astype(np.float64)

                    fileShortName = filename[:-4]
                    sig = sig.flatten()
                    # print(fileShortName)
                    AllSig[fileShortName] = sig
        return AllSig
    def saveListH5(self, fileName, TrainList, TrainLabelList, RawTrainList, TestList, TestLabelList, RawTestList):
        file = h5py.File(fileName, 'w')
        print(TrainList.shape, TrainLabelList.shape, TestList.shape, TestLabelList.shape)
        file.create_dataset('TrainList', data=TrainList)
        file.create_dataset('RawTrainList', data=RawTrainList)
        file.create_dataset('TrainLabelList', data=TrainLabelList)
        file.create_dataset('TestList', data=TestList)
        file.create_dataset('RawTestList', data=RawTestList)
        file.create_dataset('TestLabelList', data=TestLabelList)
        file.close()

    def csvfileread(self, path):  # 读取.csv文件
        labelDict = {}
        csvReader = csv.reader(open(path))
        csvContext = []
        for line in csvReader:
            fileName = line[0]
            label = line[1]
            labelDict[fileName] = label
        return labelDict  # return a label dict

    def labelToNum(self, labelDict):  # 把label转换微数字
        for i in labelDict:
            if labelDict[i] == 'N':
                labelDict[i] = 0
            elif labelDict[i] == 'A':
                labelDict[i] = 1
            elif labelDict[i] == 'O':
                labelDict[i] = 2
            else:
                labelDict[i] = 3
        return labelDict

    def getLabel(self, rootdir="../Challenge2017"):
        labelDict = self.csvfileread(rootdir + '/' + 'REFERENCE.csv')
        labelDict = self.labelToNum(labelDict)
        return labelDict

    def listExclude(self, List, excludeList):
        count = 0
        for i in excludeList:
            if i in List:
                popIndex = List.index(i)
                List.pop(popIndex)
                count += 1
                # print(count)
        return List
    def splitTrainTestNew(self, AllLabel, train_size=0.5):
        from sklearn.utils import shuffle
        trainNameList = []
        testNameList = []

        # shuffle and split Namelist for each class (4 class)
        def shuffleList(List):
            shuffle_List = shuffle(List)
            len_all = len(shuffle_List)
            split_number = int(len_all * train_size)
            # shuffle list
            trainList = shuffle_List[0:split_number]
            testList = shuffle_List[split_number:]
            print(len(trainList), len(testList))
            return trainList, testList

        classNameList0 = []
        classNameList1 = []
        classNameList2 = []
        classNameList3 = []
        for i in AllLabel:
            print(i, AllLabel[i])
            if (0 == AllLabel[i]):
                classNameList0.append(i)
            if (1 == AllLabel[i]):
                classNameList1.append(i)
            if (2 == AllLabel[i]):
                classNameList2.append(i)
            if (3 == AllLabel[i]):
                classNameList3.append(i)
        print(len(classNameList0), len(classNameList1), len(classNameList2), len(classNameList3))
        trainList, testList = shuffleList(classNameList0)
        trainNameList.extend(trainList)
        testNameList.extend(testList)

        trainList, testList = shuffleList(classNameList1)
        trainNameList.extend(trainList)
        testNameList.extend(testList)

        trainList, testList = shuffleList(classNameList2)
        trainNameList.extend(trainList)
        testNameList.extend(testList)

        trainList, testList = shuffleList(classNameList3)
        trainNameList.extend(trainList)
        testNameList.extend(testList)

        trainNameList = shuffle(trainNameList)
        testNameList = shuffle(testNameList)

        return trainNameList, testNameList

    ###RawData########################################################
    # using in getRawFilterData
    def filterFun(self, sig, fs=300, lowPass=1, highPass=150):
        f = FIRFilters()
        filtecg = f.highpass(sig, fs, lowPass)
        filtecg = f.lowpass(filtecg, fs, highPass)
        return filtecg

    def getRawFilterArray(self, AllSig, AllLabel, fileNameList, cutLength=1 * 300,
                          mutiplesNumber=1):  # 每个mat文件只取一个数据，数据量少的多取，目的是平衡数据
        myData = []
        myLabel = []
        for fileName in fileNameList:
            sig = AllSig[fileName]
            sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
            sig=sig[self.fs:]
            lenSig = len(sig)
            if (lenSig < (cutLength + self.fs + 1)):
                continue
            # Normolize,must be doing after segment
            if AllLabel[fileName] == 0:
                for j in range(2 * mutiplesNumber):  # 1
                    sig_segment = self.GetRawFilterSegmentArray_MaxPointJoint(sig, self.fs, cutLength, segNum=4)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 1:
                for j in range(14 * mutiplesNumber):  # 7
                    sig_segment = self.GetRawFilterSegmentArray_MaxPointJoint(sig, self.fs, cutLength, segNum=4)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 2:
                for j in range(4 * mutiplesNumber):  # 2
                    sig_segment = self.GetRawFilterSegmentArray_MaxPointJoint(sig, self.fs, cutLength, segNum=4)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 3:
                for j in range(35 * mutiplesNumber):  # 120
                    sig_segment = self.GetRawFilterSegmentArray_MaxPointJoint(sig, self.fs, cutLength, segNum=4)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
                    # print(sig_segment.shape)
        myData = np.asarray(myData)
        myLabel = np.asarray(myLabel)
        return myData, myLabel

    def GetRawFilterSegmentArray_MaxPointJoint(self, sig, fs=300, segLength=3 * 300, segNum=4):
        randomSegment_Array = []
        segmentAll = []
        sig_cutLastsection = len(sig) - segLength - fs  # avoid segment exceed sig max length
        for k in range(segNum):
            randomPosition = random.randint(0, sig_cutLastsection - 1)
            start_pos = randomPosition
            end_pos = start_pos + segLength + fs
            if k == 0:
                start_pos = start_pos  # first start position don't need change
                #start_pos = self.getStartNextMaxValuePositon(sig, fs, start_pos)
            else:
                start_pos = self.getStartNextMaxValuePositon(sig, fs, start_pos)
            if k == segNum - 1:
                end_pos = end_pos  # last end position extend 1 second ,For simplifying caculation
            else:
                end_pos = self.getEndLastMaxValuePositon(sig, fs, end_pos)
            sig_segment = sig[start_pos:end_pos]
            f = FIRFilters()
            sig_segment = f.normalize_DivMax(sig_segment)
            segmentAll[len(segmentAll):len(sig_segment)] = sig_segment
        if (len(segmentAll) > segLength * segNum):
            segmentAll = segmentAll[:segLength * segNum]

        segmentAll = np.asarray(segmentAll)
        # second normalize
        f = FIRFilters()
        segmentAll = f.normalize(segmentAll)

        sig_filter1 = self.filterFun(segmentAll, fs, 1, 70)
        sig_filter2 = self.filterFun(segmentAll, fs, 5, 90)
        sig_filter3 = self.filterFun(segmentAll, fs, 20, 120)
        randomSegment_Array.append(segmentAll)
        randomSegment_Array.append(sig_filter1)
        randomSegment_Array.append(sig_filter2)
        randomSegment_Array.append(sig_filter3)
        randomSegment_Array = np.asarray(randomSegment_Array)  # get 4*6s array
        # print(randomSegment_Array.shape)
        return randomSegment_Array

    def GetRawFilterSegment_MaxPointJoint(self, sig, fs=300, segLength=3 * 300, segNum=4):
        randomSegment_Array = []
        segmentAll = []
        import time
        sig_cutLastsection = len(sig) - segLength - fs  # avoid segment exceed sig max length
        # print(len(sig),sig_cutLastsection)
        # if(sig_cutLastsection<0 and sig>segLength):#ex:sig len =2000,segLength=1800
        for k in range(segNum):
            randomPosition = random.randint(0, sig_cutLastsection - 1)
            start_pos = randomPosition
            end_pos = start_pos + segLength + fs
            if k == 0:
                start_pos = start_pos  # first start position don't need change
            else:
                start_pos = self.getStartNextMaxValuePositon(sig, self.fs, start_pos)
            if k == segNum - 1:
                end_pos = end_pos  # last end position extend 1 second ,For simplifying caculation
            else:
                end_pos = self.getEndLastMaxValuePositon(sig, self.fs, end_pos)
            sig_segment = sig[start_pos:end_pos]
            segmentAll[len(segmentAll):len(sig_segment)] = sig_segment
        if (len(segmentAll) > segLength * segNum):
            segmentAll = segmentAll[:segLength * segNum]
        segmentAll = np.asarray(segmentAll)
        # this is for clean data
        f = FIRFilters()
        sig_filter1 = self.filterFun(segmentAll, self.fs, 1, 70)
        randomSegment_Array.append(sig_filter1)
        randomSegment_Array = np.asarray(randomSegment_Array)  # get 4*6s array
        # print(randomSegment_Array.shape)
        return randomSegment_Array

    def getOneRawFilterArray(self, AllSig, AllLabel, fileNameList, cutLength=1 * 300,
                             mutiplesNumber=1):  # 每个mat文件只取一个数据，数据量少的多取，目的是平衡数据
        myData = []
        myLabel = []
        for fileName in fileNameList:
            sig = AllSig[fileName]
            sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
            # sig = sig[self.fs:]  # first second is not reliable
            # print(fileName)
            # print(sig.shape)
            if AllLabel[fileName] == 0:
                for j in range(2 * mutiplesNumber):  # 1
                    sig_segment = self.GetRawFilterSegment_MaxPointJoint(sig, self.fs, cutLength, segNum=4)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 1:
                for j in range(14 * mutiplesNumber):  # 7
                    sig_segment = self.GetRawFilterSegment_MaxPointJoint(sig, self.fs, cutLength, segNum=4)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 2:
                for j in range(4 * mutiplesNumber):  # 2
                    sig_segment = self.GetRawFilterSegment_MaxPointJoint(sig, self.fs, cutLength, segNum=4)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 3:
                for j in range(35 * mutiplesNumber):  # 120
                    sig_segment = self.GetRawFilterSegment_MaxPointJoint(sig, self.fs, cutLength, segNum=4)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
                    # print(sig_segment.shape)
        myData = np.asarray(myData)
        myLabel = np.asarray(myLabel)
        return myData, myLabel

    def getStartNextMaxValuePositon(self, signal, fs, startPosition):
        absolute_position = startPosition + np.argmax(signal[startPosition:startPosition + fs])
        # relative_position = np.argmax(signal[startPosition:startPosition + fs])
        return absolute_position

    def getEndLastMaxValuePositon(self, signal, fs, endPosition):
        absolute_position = endPosition + np.argmax(signal[endPosition:endPosition + fs])
        # relative_position = np.argmax(signal[endPosition:endPosition + fs])
        return absolute_position

    # modify start

    def getRawFilterArray_modify(self, AllSig, AllLabel, fileNameList, cutLength=1 * 300,
                                 mutiplesNumber=1, segNum=4):  # 每个mat文件只取一个数据，数据量少的多取，目的是平衡数据
        myData = []
        myLabel = []
        for fileName in fileNameList:
            sig = AllSig[fileName]
            sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
            lenSig = len(sig)
            if (lenSig < (cutLength + 2*self.fs + 1)):
                continue
            if (len(sig) > 26 * self.fs):
                mutiplesNumber = 4
            else:
                mutiplesNumber=1
            if AllLabel[fileName] == 0:
                for j in range(1 * mutiplesNumber):  # 1
                    sig_segment = self.GetRawFilterSegmentArray_Modify(sig, self.fs, cutLength, segNum=segNum)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 1:
                for j in range(7 * mutiplesNumber):  # 7
                    sig_segment = self.GetRawFilterSegmentArray_Modify(sig, self.fs, cutLength, segNum=segNum)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 2:
                for j in range(2 * mutiplesNumber):  # 2
                    sig_segment = self.GetRawFilterSegmentArray_Modify(sig, self.fs, cutLength, segNum=segNum)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 3:
                for j in range(17 * mutiplesNumber):  # 120
                    sig_segment = self.GetRawFilterSegmentArray_Modify(sig, self.fs, cutLength, segNum=segNum)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
                    # print(sig_segment.shape)
        myData = np.asarray(myData)
        myLabel = np.asarray(myLabel)
        return myData, myLabel

    def GetRawFilterSegmentArray_Modify(self, sig, fs=300, segLength=3 * 300, segNum=4):
        randomSegment_Array = []
        segmentAll = []
        sig=sig[fs:]#first second is not realiable
        if (len(sig) > 25 * self.fs):
            allLength = segLength * segNum
            sig_cutLastsection = len(sig) - allLength
            randomPosition = random.randint(0, sig_cutLastsection - 1)
            segmentAll = sig[randomPosition:randomPosition + allLength]
        else:
            sig_cutLastsection = len(sig) - segLength - fs  # avoid segment exceed sig max length
            for k in range(segNum):
                randomPosition = random.randint(0, sig_cutLastsection - 1)
                start_pos = randomPosition
                end_pos = start_pos + segLength + fs
                if k == 0:
                    start_pos = start_pos  # first start position don't need change
                    #start_pos = self.getStartNextMaxValuePositon(sig, self.fs, start_pos)
                else:
                    start_pos = self.getStartNextMaxValuePositon(sig, self.fs, start_pos)
                if k == segNum - 1:
                    end_pos = end_pos  # last end position extend 1 second ,For simplifying caculation
                else:
                    end_pos = self.getEndLastMaxValuePositon(sig, self.fs, end_pos)
                sig_segment = sig[start_pos:end_pos]
                # first normalize
                f = FIRFilters()
                sig_segment = f.normalize_DivMax(sig_segment)
                segmentAll[len(segmentAll):len(sig_segment)] = sig_segment
            if (len(segmentAll) > segLength * segNum):
                segmentAll = segmentAll[:segLength * segNum]
            segmentAll = np.asarray(segmentAll)
        # seconds normalize
        f = FIRFilters()
        segmentAll = f.normalize(segmentAll)
        #print("segmentAll shape",segmentAll.shape)
        sig_filter1 = self.filterFun(segmentAll, self.fs, 1, 70)
        sig_filter2 = self.filterFun(segmentAll, self.fs, 5, 90)
        sig_filter3 = self.filterFun(segmentAll, self.fs, 20, 120)
        randomSegment_Array.append(segmentAll)
        randomSegment_Array.append(sig_filter1)
        randomSegment_Array.append(sig_filter2)
        randomSegment_Array.append(sig_filter3)
        randomSegment_Array = np.asarray(randomSegment_Array)  # get 4*6s array
        # print(randomSegment_Array.shape)
        return randomSegment_Array
        # modify end


        # modify version 2 start

    def getRawFilterArray_modify2(self, AllSig, AllLabel, fileNameList, cutLength=1 * 300,
                                  mutiplesNumber=1, segNum=4):  # 每个mat文件只取一个数据，数据量少的多取，目的是平衡数据
        myData = []
        myLabel = []
        for fileName in fileNameList:
            sig = AllSig[fileName]
            sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
            lenSig = len(sig)
            if (lenSig < (cutLength + 2*self.fs + 1)):
                continue
            if (len(sig) > 26 * self.fs):
                mutiplesNumber = 3
            else:
                mutiplesNumber=1
            # sig = f.normalize(sig)
            if AllLabel[fileName] == 0:
                for j in range(1 * mutiplesNumber):  # 1
                    sig_segment = self.GetRawFilterSegmentArray_Modify2(sig, self.fs, cutLength, segNum=segNum)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 1:
                for j in range(7 * mutiplesNumber):  # 7
                    sig_segment = self.GetRawFilterSegmentArray_Modify2(sig, self.fs, cutLength, segNum=segNum)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 2:
                for j in range(2 * mutiplesNumber):  # 2
                    sig_segment = self.GetRawFilterSegmentArray_Modify2(sig, self.fs, cutLength, segNum=segNum)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 3:
                for j in range(17 * mutiplesNumber):  # 120
                    sig_segment = self.GetRawFilterSegmentArray_Modify2(sig, self.fs, cutLength, segNum=segNum)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
                    # print(sig_segment.shape)
        myData = np.asarray(myData)
        myLabel = np.asarray(myLabel)
        return myData, myLabel

    def GetRawFilterSegmentArray_Modify2(self, sig, fs=300, segLength=3 * 300, segNum=4):
        randomSegment_Array = []
        segmentAll = []
        if (len(sig) > 25 * self.fs):
            allLength = segLength * segNum
            sig_cutLastsection = len(sig) - allLength - fs
            randomPosition = random.randint(0, sig_cutLastsection - 1)
            start_pos=self.getStartNextMaxValuePositon(sig, fs, randomPosition)
            segmentAll = sig[start_pos:start_pos + allLength]
        else:
            sig_cutLastsection = len(sig) - segLength - fs  # avoid segment exceed sig max length
            for k in range(segNum):
                #step = int(round(sig_cutLastsection / segNum))
                #randomPosition=random.randint(k*step,(k+1)*step)
                randomPosition = random.randint(0, sig_cutLastsection - 1)
                start_pos = randomPosition
                end_pos = start_pos + segLength + fs
                if k == 0:
                    #start_pos = start_pos  # first start position don't need change
                    start_pos = self.getStartNextMaxValuePositon(sig, self.fs, start_pos)
                else:
                    start_pos = self.getStartNextMaxValuePositon(sig, self.fs, start_pos)
                if k == segNum - 1:
                    end_pos = end_pos  # last end position extend 1 second ,For simplifying caculation
                else:
                    end_pos = self.getEndLastMaxValuePositon(sig, self.fs, end_pos)
                sig_segment = sig[start_pos:end_pos]
                # first normalize
                f = FIRFilters()
                sig_segment = f.normalize_DivMax(sig_segment)
                segmentAll[len(segmentAll):len(sig_segment)] = sig_segment
            if (len(segmentAll) > segLength * segNum):
                segmentAll = segmentAll[:segLength * segNum]
            segmentAll = np.asarray(segmentAll)
        # seconds normalize
        f = FIRFilters()
        segmentAll = f.normalize(segmentAll)
        sig_filter1 = self.filterFun(segmentAll, self.fs, 1, 70)
        sig_filter2 = self.filterFun(segmentAll, self.fs, 5, 90)
        sig_filter3 = self.filterFun(segmentAll, self.fs, 20, 120)
        randomSegment_Array.append(segmentAll)
        randomSegment_Array.append(sig_filter1)
        randomSegment_Array.append(sig_filter2)
        randomSegment_Array.append(sig_filter3)
        randomSegment_Array = np.asarray(randomSegment_Array)  # get 4*6s array
        # print(randomSegment_Array.shape)
        return randomSegment_Array
        # modify2 end
    def readPlotData(self, inputNum=1, inputTickdim=1200, inputFeatureDim=20, outputClass=4,
                     fileName='TrainTestRaw.h5'):
        file = h5py.File(fileName, 'r')
        testData = file['TestList']
        testLable = file['TestLabelList']

        def PreProcessing(data, dataLabel):
            print(np.asarray(data).shape)
            numData = len(data)
            data = np.swapaxes(data, 1, 2)
            data = np.asarray(data).reshape((numData, inputTickdim, inputFeatureDim))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], 1))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable

        print("Load from hdf5 file,the shap is ")
        testData, testLable = PreProcessing(testData, testLable)

        # shuffle the data

        testData, testLable = shuffle(testData, testLable, random_state=0)
        # generate valide data
        numSplit = int(round(0.8 * len(testData)))

        X_test = testData[:numSplit]
        y_test = testLable[:numSplit]

        # reshape data

        X_test = np.reshape(X_test, (-1, inputTickdim, inputFeatureDim))

        Y_test = to_categorical(y_test, outputClass)
        print("The output shape is: ")
        print('X_test shape:', X_test.shape)
        print('y_test target', y_test.shape)
        print('Y_test target', Y_test.shape)
        file.close()
        return testData, testLable, X_test, Y_test

    def prepareRawFilterArrayData(self, trainData, trainLabel, testData, testLabel, inputTickdim=1200,
                                  inputFeatureDim=20, outputClass=4):
        numTrain = len(trainData)

        def PreProcessing(data, dataLabel):
            print(np.asarray(data).shape)
            numData = len(data)
            data = np.swapaxes(data, 1, 2)
            data = np.asarray(data).reshape((numData, inputTickdim, inputFeatureDim))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], 1))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable

        print("prepare Data ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)
        testData, testLable = PreProcessing(testData, testLabel)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
        testData, testLable = shuffle(testData, testLable, random_state=0)
        # generate valide data
        numSplit = int(round(1 * len(testData)))

        X_test = testData[:numSplit]
        y_test = testLable[:numSplit]
        X_valid = testData[numSplit:]
        y_valid = testLable[numSplit:]

        X_train = trainData
        y_train = trainLabel

        # reshape data
        X_train = np.reshape(X_train, (-1, inputTickdim, inputFeatureDim))
        X_valid = np.reshape(X_valid, (-1, inputTickdim, inputFeatureDim))
        X_test = np.reshape(X_test, (-1, inputTickdim, inputFeatureDim))

        Y_train = to_categorical(y_train, outputClass)
        Y_valid = to_categorical(y_valid, outputClass)
        Y_test = to_categorical(y_test, outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

    def savePredictAndTarget(self, y_test, y_pred, saveName=""):
        print("predict shape", y_pred.shape)
        directoryName = 'PredictResult'
        print("save feature Data")

        print("save predictLabel Data")
        np.save(directoryName + '/' + saveName + "predict.npy", np.asarray(y_pred))
        print("save targetLabel Data")
        np.save(directoryName + '/' + saveName + "target.npy", np.asarray(y_test))

    def readRawArrayData(self, inputNum=1, inputTickdim=1200, inputFeatureDim=20, outputClass=4,
                         fileName='TrainTestRaw.h5'):
        file = h5py.File(fileName, 'r')
        trainData = file['TrainList']
        trainLabel = file['TrainLabelList']
        testData = file['TestList']
        testLable = file['TestLabelList']
        numTrain = len(trainData)

        def PreProcessing(data, dataLabel):
            print(np.asarray(data).shape)
            numData = len(data)
            data = np.swapaxes(data, 1, 2)
            data = np.asarray(data).reshape((numData, inputTickdim, inputFeatureDim))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], 1))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable

        print("Load from hdf5 file,the shap is ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)
        testData, testLable = PreProcessing(testData, testLable)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
        testData, testLable = shuffle(testData, testLable, random_state=0)
        # generate valide data
        numSplit = int(round(0.8 * len(testData)))
        # X_train = trainData[:numSplit]
        # y_train = trainLabel[:numSplit]
        # X_valid = trainData[numSplit:numTrain]
        # y_valid = trainLabel[numSplit:numTrain]
        #
        # X_test = testData
        # y_test = testLable
        X_test = testData[:numSplit]
        y_test = testLable[:numSplit]
        X_valid = testData[numSplit:]
        y_valid = testLable[numSplit:]
        y_valid = testLable[numSplit:]

        X_train = trainData
        y_train = trainLabel

        # reshape data
        X_train = np.reshape(X_train, (-1, inputTickdim, inputFeatureDim))
        X_valid = np.reshape(X_valid, (-1, inputTickdim, inputFeatureDim))
        X_test = np.reshape(X_test, (-1, inputTickdim, inputFeatureDim))

        Y_train = to_categorical(y_train, outputClass)
        Y_valid = to_categorical(y_valid, outputClass)
        Y_test = to_categorical(y_test, outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        file.close()
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
    #####FBANK FUNCTION###############################################
    def melSpectrogram(self, sig_segment, fs, winlen=0.05, winstep=0.03):

        # mfcc_feet, energy = logfbank(sig_segment, samplerate=fs, nfilt=40, winlen=winlen, winstep=winstep, lowfreq=5,
        #                              highfreq=120)
        mfcc_feet, energy = logfbank(sig_segment, samplerate=fs, nfilt=26, winlen=winlen, winstep=winstep)
        return mfcc_feet

    def fromRawArrayToFbank(self, rawArray, fs=300):
        fbankArray = []
        for rawSegment in rawArray:
            logFbank = self.melSpectrogram(rawSegment, self.fs)
            # print(logFbank.shape)
            fbankArray.append(logFbank)
        fbankArray = np.asarray(fbankArray)
        return fbankArray

    def GetFbankSegmentArray_MaxPointJoint(self, sig, fs=300, segLength=3 * 300, segNum=4):
        randomSegment_Array = []
        segmentAll = []
        sig_cutLastsection = len(sig) - segLength - fs  # avoid segment exceed sig max length

        for k in range(segNum):
            randomPosition = random.randint(0, sig_cutLastsection - 1)
            start_pos = randomPosition
            end_pos = start_pos + segLength + fs
            if k == 0:
                start_pos = start_pos  # first start position don't need change
            else:
                start_pos = self.getStartNextMaxValuePositon(sig, self.fs, start_pos)
            if k == segNum - 1:
                end_pos = end_pos  # last end position extend 1 second ,For simplifying caculation
            else:
                end_pos = self.getEndLastMaxValuePositon(sig, self.fs, end_pos)
            sig_segment = sig[start_pos:end_pos]
            segmentAll[len(segmentAll):len(sig_segment)] = sig_segment
        if (len(segmentAll) > segLength * segNum):
            segmentAll = segmentAll[:segLength * segNum]
        segmentAll = np.asarray(segmentAll)
        sig_filter1 = self.filterFun(segmentAll, self.fs, 1, 70)
        randomSegment_Array.append(segmentAll)
        randomSegment_Array.append(sig_filter1)
        randomSegment_Array = np.asarray(randomSegment_Array)  # get 4*6s array
        # print(randomSegment_Array.shape)
        return randomSegment_Array

    def getFbankArrayData(self, AllSig, AllLabel, fileNameList, cutLength=6 * 300,
                          mutiplesNumber=1):  # 每个mat文件只取一个数据，数据量少的多取，目的是平衡数据
        myData = []
        myLabel = []
        for fileName in fileNameList:
            sig = AllSig[fileName]
            sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
            # add _filter
            f = FIRFilters()
            filtecg = f.highpass(sig, 300, 1)
            filtecg = f.lowpass(filtecg, 300, 120)
            ###################
            sig = filtecg[self.fs:]  # first second is not reliable
            sigLen = len(sig)
            # print(fileName)
            if AllLabel[fileName] == 0:
                for j in range(1 * mutiplesNumber):  # 1 the number of geting segment from a signal
                    rawSegmentArray = self.GetFbankSegmentArray_MaxPointJoint(sig, self.fs, cutLength, 4)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank(
                        rawSegmentArray)  # get 10*39*40 number * step * specvector
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])
                    # print(fbankSegmentArray.shape)
            elif AllLabel[fileName] == 1:
                for j in range(7 * mutiplesNumber):  # 7
                    rawSegmentArray = self.GetFbankSegmentArray_MaxPointJoint(sig, self.fs, cutLength, 4)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank(
                        rawSegmentArray)  # get 10*39*40 number * step * specvector
                    # print(fbankSegmentArray.shape)
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 2:
                for j in range(2 * mutiplesNumber):  # 2
                    rawSegmentArray = self.GetFbankSegmentArray_MaxPointJoint(sig, self.fs, cutLength, 4)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank(
                        rawSegmentArray)  # get 10*39*40 number * step * specvector
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 3:
                for j in range(18 * mutiplesNumber):  # 120
                    rawSegmentArray = self.GetFbankSegmentArray_MaxPointJoint(sig, self.fs, cutLength, 4)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank(
                        rawSegmentArray)  # get 10*39*40 number * step * specvector
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])
        myData = np.asarray(myData)
        myLabel = np.asarray(myLabel)
        return myData, myLabel

    def readFbankArrayData(self, inputNum=2, img_rows=39, img_cols=40, outputClass=4,
                           fileName='TrainTestDataFbankArray.h5'):
        file = h5py.File(fileName, 'r')
        trainData = file['TrainList']
        trainLabel = file['TrainLabelList']
        testData = file['TestList']
        testLable = file['TestLabelList']
        numTrain = len(trainData)

        def PreProcessing(data, dataLabel):
            print(np.asarray(data).shape)
            data = np.asarray(data).reshape((-1, inputNum, img_rows, img_cols))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], 1))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable

        print("Load from hdf5 file,the shap is ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)
        testData, testLable = PreProcessing(testData, testLable)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
        testData, testLable = shuffle(testData, testLable, random_state=0)
        # generate valide data
        numSplit = int(round(0.8 * len(testData)))
        X_test = testData[:numSplit]
        y_test = testLable[:numSplit]
        X_valid = testData[numSplit:]
        y_valid = testLable[numSplit:]

        X_train = trainData
        y_train = trainLabel

        # reshape data
        X_train = np.reshape(X_train, (-1, inputNum, img_rows, img_cols))
        X_valid = np.reshape(X_valid, (-1, inputNum, img_rows, img_cols))
        X_test = np.reshape(X_test, (-1, inputNum, img_rows, img_cols))

        Y_train = to_categorical(y_train, outputClass)
        Y_valid = to_categorical(y_valid, outputClass)
        Y_test = to_categorical(y_test, outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        file.close()
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

    def prepareFbankArrayData(self, trainData, trainLabel, testData, testLabel, inputNum=2, img_rows=39, img_cols=40,
                              outputClass=4,
                              fileName='TrainTestDataFbankArray.h5'):

        def PreProcessing(data, dataLabel):
            print(np.asarray(data).shape)
            data = np.asarray(data).reshape((-1, inputNum, img_rows, img_cols))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], 1))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable

        print("Load from hdf5 file,the shap is ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)
        testData, testLable = PreProcessing(testData, testLabel)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
        testData, testLable = shuffle(testData, testLable, random_state=0)
        # generate valide data
        numSplit = int(round(0.8 * len(testData)))
        X_test = testData[:numSplit]
        y_test = testLable[:numSplit]
        X_valid = testData[numSplit:]
        y_valid = testLable[numSplit:]

        X_train = trainData
        y_train = trainLabel

        # reshape data
        X_train = np.reshape(X_train, (-1, inputNum, img_rows, img_cols))
        X_valid = np.reshape(X_valid, (-1, inputNum, img_rows, img_cols))
        X_test = np.reshape(X_test, (-1, inputNum, img_rows, img_cols))

        Y_train = to_categorical(y_train, outputClass)
        Y_valid = to_categorical(y_valid, outputClass)
        Y_test = to_categorical(y_test, outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        #file.close()
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

    def saveEpochWeight(self, SaveName, model, i):
        import os
        import logging
        if os.path.isdir(SaveName) == False:
            os.mkdir(SaveName)
        json_string = model.to_json()
        open(SaveName + '/' + SaveName + '.json', 'w').write(json_string)
        model.save_weights(SaveName + '/-' + str(i) + '-' '.h5')

    def saveEpochLog(self, SaveName, i, allScore):
        import os
        import logging
        if os.path.isdir(SaveName) == False:
            os.mkdir(SaveName)
        if (len(allScore) > 0):
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                                filename=SaveName + '/' + 'ExperimentRecord.log', filemode='a')
            logging.info('All score' + str(allScore[0]))
            logging.info('Nor score' + str(allScore[1]))
            logging.info('AF score' + str(allScore[2]))
            logging.info('Other score' + str(allScore[3]))
            logging.info('Noise score' + str(allScore[4]))
            logging.info('\n')


class EcgEngryLib:
    fs = 300
    input_num = 100
    cutLength = 6 * fs

    def filterFun(self, sig, fs=300, lowPass=1, highPass=150):
        f = FIRFilters()
        filtecg = f.highpass(sig, fs, lowPass)
        filtecg = f.lowpass(filtecg, fs, highPass)
        return filtecg

    def getStartNextMaxValuePositon(self, signal, fs, startPosition):
        absolute_position = startPosition + np.argmax(signal[startPosition:startPosition + fs])
        # relative_position = np.argmax(signal[startPosition:startPosition + fs])
        return absolute_position

    def getEndLastMaxValuePositon(self, signal, fs, endPosition):
        absolute_position = endPosition + np.argmax(signal[endPosition:endPosition + fs])
        # relative_position = np.argmax(signal[endPosition:endPosition + fs])
        return absolute_position


    def write_answer(self, filename, result, resultfile="answers.txt"):
        fo = open(resultfile, 'a')
        fo.write(str(filename) + "," + str(result) + "\n")
        fo.close()

    def oneHotToNumber(self, y_vector):
        predict = []
        for itr in y_vector:
            predict.append(np.argmax(itr))
        return np.asarray(predict)

    def vote(self, y_predict):
        voteArray = [0, 0, 0, 0]
        for i in y_predict:
            voteArray[i] = voteArray[i] + 1
        # print(voteArray)
        voteArray = np.asarray(voteArray)
        # print(voteArray,voteArray.shape)
        para = (sum(voteArray) / 12)
        if (np.argmax(voteArray) == 0):
            voteArray[2] = para * voteArray[2]  # other multi 1.5,multi voteNum/10
            # if (np.argmax(voteArray) == 3):
            #     voteArray[0] = voteArray[0]/1.5 #lstm
            # voteArray[2] = voteArray[2]/1.5 #cnn
        return np.argmax(voteArray)

    def getModelInput(self, fileName, AllSig, voteNum, cutLength=6*300):
        model_input = []
        rootdir = "../Challenge2017/"
        # sig, fields = wfdb.srdsamp(rootdir+str(fileName))
        sig = AllSig[fileName]
        sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
        for i in range(voteNum):
            cef = ChallengeEcgFun()
            oneInput = cef.GetRawFilterSegmentArray_MaxPointJoint(sig, self.fs, cutLength, segNum=4)
            # print(oneInput.shape)
            model_input.append(oneInput)
        model_input = np.asarray(model_input)
        model_input = np.swapaxes(model_input, 1, 2)
        return model_input

    def getFileTestF1score(self, model, AllSig, AllLabel, fileNameList, voteNum, cutLength=6 * 300, isLoadWeights=False):
        from keras.models import model_from_json
        import sklearn.metrics as metrics
        import datetime
        startTime = datetime.datetime.now()
        targetLabel = []
        predictLabel = []
        if (isLoadWeights == True):
            print('Build model...')
            saveName = "rawFilterArray"
            model = model_from_json(open(saveName + '.json').read())
            model.load_weights(saveName + '.h5')
        for fileName in fileNameList:
            if (len(AllSig[fileName]) < cutLength + self.fs + 1):
                print(fileName + "is shorter than cutLength")
                continue
            model_input = self.getModelInput(fileName, AllSig, voteNum, cutLength=cutLength)
            y_pred = model.predict(model_input, batch_size=voteNum)
            y_predict = self.oneHotToNumber(y_pred)
            average_prediction = self.vote(y_predict)
            # print(average_prediction)
            predictLabel.append(average_prediction)
            targetLabel.append(AllLabel[fileName])
        y_test = np.asarray(targetLabel)
        y_predict = np.asarray(predictLabel)
        # print(y_test,y_predict)
        import sklearn.metrics as metrics
        # y_test = [2, 0, 2, 2, 0, 1]
        # y_predict = [0, 0, 2, 2, 0, 2]
        # print(y_test, y_predict)
        # cm = metrics.confusion_matrix(y_test, y_pred=y_predict, labels=[0, 1, 2, 3])
        cm = metrics.confusion_matrix(y_test, y_pred=y_predict)

        print(cm)
        cm_nor = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm_nor.shape[0]):
            for j in range(cm_nor.shape[1]):
                cm_nor[i][j] = float('%.2f' % cm_nor[i][j])
        print(cm_nor)
        cem = ChallengeEcgMetric()
        all_fscore = cem.f_score(cm_nor)
        endTime = datetime.datetime.now()
        print("all_fscore,three_fscore", all_fscore[0], all_fscore[1])
        print("Usin time:", (endTime - startTime).seconds)
        return all_fscore

    def getModelInput_Modify(self, fileName, AllSig, voteNum, cutLength=6 * 300, segNum=4):

        model_input = []
        sig = AllSig[fileName]
        sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
        for i in range(voteNum):
            cef = ChallengeEcgFun()
            oneInput = cef.GetRawFilterSegmentArray_Modify(sig, self.fs, cutLength, segNum=segNum)
            # print(oneInput.shape)
            model_input.append(oneInput)
        model_input = np.asarray(model_input)
        model_input = np.swapaxes(model_input, 1, 2)
        return model_input

    def getFileTestF1score_Modify(self, model, AllSig, AllLabel, fileNameList, voteNum, cutLength=6 * 300,
                                  isLoadWeights=False, segNum=4):
        from keras.models import model_from_json
        import sklearn.metrics as metrics
        import datetime
        startTime = datetime.datetime.now()
        targetLabel = []
        predictLabel = []
        if (isLoadWeights == True):
            print('Build model...')
            saveName = "rawFilterArray"
            model = model_from_json(open(saveName + '.json').read())
            model.load_weights(saveName + '.h5')
        for fileName in fileNameList:
            if (len(AllSig[fileName]) < cutLength + 2*self.fs + 1):
                print(fileName + "is shorter than cutLength")
                continue
            model_input = self.getModelInput_Modify(fileName, AllSig, voteNum, cutLength=cutLength, segNum=segNum)
            y_pred = model.predict(model_input, batch_size=voteNum)
            y_predict = self.oneHotToNumber(y_pred)
            average_prediction = self.vote(y_predict)
            # print(average_prediction)
            predictLabel.append(average_prediction)
            targetLabel.append(AllLabel[fileName])
        y_test = np.asarray(targetLabel)
        y_predict = np.asarray(predictLabel)

        import sklearn.metrics as metrics
        cm = metrics.confusion_matrix(y_test, y_pred=y_predict)

        print(cm)
        cm_nor = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm_nor.shape[0]):
            for j in range(cm_nor.shape[1]):
                cm_nor[i][j] = float('%.2f' % cm_nor[i][j])
        print(cm_nor)
        cem = ChallengeEcgMetric()
        all_fscore = cem.f_score(cm_nor)
        endTime = datetime.datetime.now()
        print("all_fscore,three_fscore", all_fscore[0], all_fscore[1])
        # print()
        print("Usin time:", (endTime - startTime).seconds)
        return all_fscore

    # modify 2 version start
    def getModelInput_Modify2(self, fileName, AllSig, voteNum, cutLength=6 * 300, segNum=4):

        model_input = []
        rootdir = "../Challenge2017/"
        # sig, fields = wfdb.srdsamp(rootdir+str(fileName))
        sig = AllSig[fileName]
        sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
        for i in range(voteNum):
            cef = ChallengeEcgFun()
            oneInput = cef.GetRawFilterSegmentArray_Modify2(sig, self.fs, cutLength, segNum=segNum)
            # print(oneInput.shape)
            model_input.append(oneInput)
        model_input = np.asarray(model_input)
        model_input = np.swapaxes(model_input, 1, 2)
        return model_input

    def getFileTestF1score_Modify2(self, model, AllSig, AllLabel, fileNameList, voteNum, cutLength=6 * 300,
                                   isLoadWeights=False, segNum=4):
        from keras.models import model_from_json
        import sklearn.metrics as metrics
        import datetime
        startTime = datetime.datetime.now()
        targetLabel = []
        predictLabel = []
        if (isLoadWeights == True):
            print('Build model...')
            saveName = "rawFilterArray"
            model = model_from_json(open(saveName + '.json').read())
            model.load_weights(saveName + '.h5')
        for fileName in fileNameList:
            if (len(AllSig[fileName]) < cutLength + 2*self.fs + 1):
                print(fileName + "is shorter than cutLength")
                continue
            model_input = self.getModelInput_Modify(fileName, AllSig, voteNum, cutLength=cutLength, segNum=segNum)
            y_pred = model.predict(model_input, batch_size=voteNum)
            y_predict = self.oneHotToNumber(y_pred)
            average_prediction = self.vote(y_predict)
            # print(average_prediction)
            predictLabel.append(average_prediction)
            targetLabel.append(AllLabel[fileName])
        y_test = np.asarray(targetLabel)
        y_predict = np.asarray(predictLabel)
        # print(y_test,y_predict)
        import sklearn.metrics as metrics
        # y_test = [2, 0, 2, 2, 0, 1]
        # y_predict = [0, 0, 2, 2, 0, 2]
        # print(y_test, y_predict)
        # cm = metrics.confusion_matrix(y_test, y_pred=y_predict, labels=[0, 1, 2, 3])
        cm = metrics.confusion_matrix(y_test, y_pred=y_predict)

        print(cm)
        cm_nor = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm_nor.shape[0]):
            for j in range(cm_nor.shape[1]):
                cm_nor[i][j] = float('%.2f' % cm_nor[i][j])
        print(cm_nor)
        cem = ChallengeEcgMetric()
        all_fscore = cem.f_score(cm_nor)
        endTime = datetime.datetime.now()
        print("all_fscore,three_fscore", all_fscore[0], all_fscore[1])
        # print()
        print("Usin time:", (endTime - startTime).seconds)
        return all_fscore
        # modify 2 end
        # def GetRawFilterSegment_MaxPointJoint(self, sig, fs=300, segLength=3 * 300, segNum=4):
        #     randomSegment_Array = []
        #     segmentAll = []
        #     sig_cutLastsection = len(sig) - segLength - fs  # avoid segment exceed sig max length
        #     for k in range(segNum):
        #         randomPosition = random.randint(0, sig_cutLastsection - 1)
        #         start_pos = randomPosition
        #         end_pos = start_pos + segLength + fs
        #         if k == 0:
        #             start_pos = start_pos  # first start position don't need change
        #         else:
        #             start_pos = self.getStartNextMaxValuePositon(sig, fs, start_pos)
        #         if k == segNum - 1:
        #             end_pos = end_pos  # last end position extend 1 second ,For simplifying caculation
        #         else:
        #             end_pos = self.getEndLastMaxValuePositon(sig, fs, end_pos)
        #         sig_segment = sig[start_pos:end_pos]
        #         segmentAll[len(segmentAll):len(sig_segment)] = sig_segment
        #     if (len(segmentAll) > segLength * segNum):
        #         segmentAll = segmentAll[:segLength * segNum]
        #     segmentAll = np.asarray(segmentAll)
        #     sig_filter1 = self.filterFun(segmentAll, fs, 1, 70)
        #     # sig_filter2 = self.filterFun(segmentAll, fs, 5, 90)
        #     # sig_filter3 = self.filterFun(segmentAll, fs, 20, 120)
        #     # randomSegment_Array.append(segmentAll)
        #     randomSegment_Array.append(sig_filter1)
        #     # randomSegment_Array.append(sig_filter2)
        #     # randomSegment_Array.append(sig_filter3)
        #     randomSegment_Array = np.asarray(randomSegment_Array)  # get 4*6s array
        #     # print(randomSegment_Array.shape)
        #     return randomSegment_Array
        # def getModelInput_one(self, fileName, AllSig, voteNum):
        #     model_input = []
        #     rootdir = "../Challenge2017/"
        #     # sig, fields = wfdb.srdsamp(rootdir+str(fileName))
        #     sig = AllSig[fileName]
        #     sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
        #     sig = sig[self.fs:]  # first second is not reliable
        #     # print(sig.shape)
        #     for i in range(voteNum):
        #         oneInput = self.GetRawFilterSegment_MaxPointJoint(sig, self.fs, self.cutLength, 4)
        #         # print(oneInput.shape)
        #         model_input.append(oneInput)
        #     model_input = np.asarray(model_input)
        #     model_input = np.swapaxes(model_input, 1, 2)
        #     return model_input
        #
        # def getFileTestF1score_one(self, model, AllSig, AllLabel, fileNameList, voteNum, isLoadWeights=False):
        #     from keras.models import model_from_json
        #     import sklearn.metrics as metrics
        #     targetLabel = []
        #     predictLabel = []
        #     if (isLoadWeights == True):
        #         print('Build model...')
        #         saveName = "rawFilterArray"
        #         model = model_from_json(open(saveName + '.json').read())
        #         model.load_weights(saveName + '.h5')
        #     for fileName in fileNameList:
        #         model_input = self.getModelInput_one(fileName, AllSig, voteNum)
        #         y_pred = model.predict(model_input, batch_size=voteNum)
        #         y_predict = self.oneHotToNumber(y_pred)
        #         average_prediction = self.vote(y_predict)
        #         # print(average_prediction)
        #         predictLabel.append(average_prediction)
        #         targetLabel.append(AllLabel[fileName])
        #     y_test = np.asarray(targetLabel)
        #     y_predict = np.asarray(predictLabel)
        #     # print(y_test,y_predict)
        #     import sklearn.metrics as metrics
        #     # y_test = [2, 0, 2, 2, 0, 1]
        #     # y_predict = [0, 0, 2, 2, 0, 2]
        #     # print(y_test, y_predict)
        #     cm = metrics.confusion_matrix(y_test, y_pred=y_predict, labels=[0, 1, 2, 3])
        #     print(cm)
        #     cm_nor = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #     for i in range(cm_nor.shape[0]):
        #         for j in range(cm_nor.shape[1]):
        #             cm_nor[i][j] = float('%.2f' % cm_nor[i][j])
        #     print(cm_nor)
        #     cem = ChallengeEcgMetric()
        #     all_fscore = cem.f_score(cm_nor)
        #     print("all_fscore,three_fscore", all_fscore[0], all_fscore[1])
        #     return all_fscore

# The class is writed for 2017 ecg challenge
class ChallengeEcgFun_Old:
    rootdir = "../Challenge2017"
    fs = 300

    def saveH5(self, fileName, TrainList, TrainLabelList, TestList, TestLabelList):
        file = h5py.File(fileName, 'w')
        print(TrainList.shape, TrainLabelList.shape, TestList.shape, TestLabelList.shape)
        file.create_dataset('TrainList', data=TrainList)
        file.create_dataset('TrainLabelList', data=TrainLabelList)
        file.create_dataset('TestList', data=TestList)
        file.create_dataset('TestLabelList', data=TestLabelList)
        # file.create_dataset('Trains1s2List', data=Trains1s2List)
        # file.create_dataset('Tests1s2List', data=Tests1s2List)
        file.close()

    def saveListH5(self, fileName, TrainList, TrainLabelList, RawTrainList, TestList, TestLabelList, RawTestList):

        file = h5py.File(fileName, 'w')
        print(TrainList.shape, TrainLabelList.shape, TestList.shape, TestLabelList.shape)
        file.create_dataset('TrainList', data=TrainList)
        file.create_dataset('RawTrainList', data=RawTrainList)
        file.create_dataset('TrainLabelList', data=TrainLabelList)
        file.create_dataset('TestList', data=TestList)
        file.create_dataset('RawTestList', data=RawTestList)
        file.create_dataset('TestLabelList', data=TestLabelList)
        # file.create_dataset('Trains1s2List', data=Trains1s2List)
        # file.create_dataset('Tests1s2List', data=Tests1s2List)
        file.close()

    def splitTrainTest(self, List, train_size=0.5):
        from sklearn.utils import shuffle
        shuffle_List = shuffle(List)
        len_all = len(shuffle_List)
        split_number = int(len_all * train_size)

        trainNameList = shuffle_List[0:split_number]
        testNameList = shuffle_List[split_number:]
        return trainNameList, testNameList

    def splitTrainTestNew(self, AllLabel, train_size=0.5):
        from sklearn.utils import shuffle
        trainNameList = []
        testNameList = []

        # shuffle and split Namelist for each class (4 class)
        def shuffleList(List):
            shuffle_List = shuffle(List)
            len_all = len(shuffle_List)
            split_number = int(len_all * train_size)
            # shuffle list
            trainList = shuffle_List[0:split_number]
            testList = shuffle_List[split_number:]
            print(len(trainList), len(testList))
            return trainList, testList

        classNameList0 = []
        classNameList1 = []
        classNameList2 = []
        classNameList3 = []
        for i in AllLabel:
            print(i, AllLabel[i])
            if (0 == AllLabel[i]):
                classNameList0.append(i)
            if (1 == AllLabel[i]):
                classNameList1.append(i)
            if (2 == AllLabel[i]):
                classNameList2.append(i)
            if (3 == AllLabel[i]):
                classNameList3.append(i)
        print(len(classNameList0), len(classNameList1), len(classNameList2), len(classNameList3))
        trainList, testList = shuffleList(classNameList0)
        trainNameList.extend(trainList)
        testNameList.extend(testList)

        trainList, testList = shuffleList(classNameList1)
        trainNameList.extend(trainList)
        testNameList.extend(testList)

        trainList, testList = shuffleList(classNameList2)
        trainNameList.extend(trainList)
        testNameList.extend(testList)

        trainList, testList = shuffleList(classNameList3)
        trainNameList.extend(trainList)
        testNameList.extend(testList)

        trainNameList = shuffle(trainNameList)
        testNameList = shuffle(testNameList)

        return trainNameList, testNameList
    def splitTrainValidTest(self, List, train_size=0.5, test_size=0.5):
        from sklearn.utils import shuffle
        shuffle_List = shuffle(List)

        len_all = len(shuffle_List)
        split_number = int(len_all * train_size)
        splist_testNumber = int(len_all * test_size)
        trainNameList = shuffle_List[0:split_number]

        testNameList = shuffle_List[split_number:split_number + splist_testNumber]
        validNameList = shuffle_List[split_number + splist_testNumber:]
        return trainNameList, testNameList, validNameList
    #return signal dict
    def getAllSig(self,rootdir="../Challenge2017"):
        AllSig={}
        for parent, dirnames, filenames in os.walk(rootdir):  # traverse the rootdir
            print(dirnames)
            for filename in filenames:

                if(filename[-3:]=='hea'):
                    # print(filename)
                    file = os.path.join(parent, filename)
                    sig, fields = wfdb.srdsamp(file[:-4])
                    #print(sig.shape)
                    fileShortName=filename[:-4]
                    AllSig[fileShortName]=sig
        return AllSig

    def csvfileread(self,path):  # 读取.csv文件
        labelDict = {}
        csvReader = csv.reader(open(path))
        csvContext = []
        for line in csvReader:
            fileName = line[0]
            label = line[1]
            labelDict[fileName] = label
        return labelDict  # return a label dict

    def labelToNum(self, labelDict):  # 把label转换微数字
        for i in labelDict:
            if labelDict[i] == 'N':
                labelDict[i] = 0
            elif labelDict[i] == 'A':
                labelDict[i] = 1
            elif labelDict[i] == 'O':
                labelDict[i] = 2
            else:
                labelDict[i] = 3
        return labelDict

    #not necessary
    def labelToVector(self,label):  # 把label转换为向量
        dicLabelToVector = {'1': [1, 0, 0, 0], '2': [0, 1, 0, 0], '3': [0, 0, 1, 0], '4': [0, 0, 0, 1]}
        labelVector = []
        for i in label:
            labelVector.append(dicLabelToVector[i])
        return np.array(labelVector)
    def getLabel(self,rootdir="../Challenge2017"):
        labelDict = self.csvfileread(rootdir + '/' + 'REFERENCE.csv')
        labelDict = self.labelToNum(labelDict)
        return labelDict
    #Huwei Write# Let's make and display a mel-scaled power (energy-squared) spectrogram
    def melSpectrogram(self, sig_segment, fs, winlen=0.2, winstep=0.1):
        #####2s 39*40
        # winlen = 0.1#
        # winstep = 0.05
        #####4s
        # winlen = 0.2  #
        # winstep = 0.1
        # mfcc_feet, energy = logfbank(sig_segment, samplerate=fs, nfilt=40, winlen=winlen, winstep=winstep, lowfreq=5,
        #                              highfreq=120)
        mfcc_feet, energy = logfbank(sig_segment, samplerate=fs, nfilt=40, winlen=winlen, winstep=winstep, highfreq=120)
        return mfcc_feet


    # Random
    def RandomCutSingle(self, AllSig, AllLabel, fileNameList, cutNum=600):  # 每个mat文件只取一个数据，数据量少的多取，目的是平衡数据
        myData = []
        myLabel = []
        for fileName in fileNameList:
            sig = AllSig[fileName]
            sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
            sig = sig[self.fs:]  # first second is not reliable
            sigLen = len(sig)
            print(fileName)
            if AllLabel[fileName] == 0:
                for j in range(4):  # 1
                    randomPosition = random.randint(0, sigLen - cutNum - 1)
                    sig_segment = sig[randomPosition:randomPosition + cutNum]
                    logFbank = self.melSpectrogram(sig_segment, 300)
                    myData.append(logFbank)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 1:
                for j in range(27):  # 7
                    randomPosition = random.randint(0, sigLen - cutNum - 1)
                    sig_segment = sig[randomPosition:randomPosition + cutNum]
                    logFbank = self.melSpectrogram(sig_segment, 300)
                    myData.append(logFbank)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 2:
                for j in range(8):  # 2
                    randomPosition = random.randint(0, sigLen - cutNum - 1)
                    sig_segment = sig[randomPosition:randomPosition + cutNum]
                    logFbank = self.melSpectrogram(sig_segment, 300)
                    myData.append(logFbank)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 3:
                for j in range(70):  # 120
                    randomPosition = random.randint(0, sigLen - cutNum - 1)
                    sig_segment = sig[randomPosition:randomPosition + cutNum]
                    logFbank = self.melSpectrogram(sig_segment, 300)
                    myData.append(logFbank)
                    myLabel.append(AllLabel[fileName])
                    print(logFbank.shape)
        myData = np.asarray(myData)
        myLabel = np.asarray(myLabel)
        return myData, myLabel

    # New random cut method :start position is max value in time
    # mutiplesNumber is number of taking segment from a signal
    def RandomCutSingle2(self, AllSig, AllLabel, fileNameList, cutNum=600,
                         mutiplesNumber=1):  # 每个mat文件只取一个数据，数据量少的多取，目的是平衡数据
        myData = []
        myLabel = []
        for fileName in fileNameList:
            sig = AllSig[fileName]
            sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
            sig = sig[self.fs:]  # first second is not reliable
            sigLen = len(sig)
            print(fileName)
            if AllLabel[fileName] == 0:
                for j in range(4 * mutiplesNumber):  # 1
                    randomPosition = random.randint(0,
                                                    sigLen - cutNum - 1 - self.fs)  # random choose a position in time in signal
                    absolute_position, relative_position = self.getNextMaxValuePositon(sig, 300, randomPosition)
                    sig_segment = sig[absolute_position:absolute_position + cutNum]
                    logFbank = self.melSpectrogram(sig_segment, 300)
                    myData.append(logFbank)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 1:
                for j in range(27 * mutiplesNumber):  # 7
                    randomPosition = random.randint(0, sigLen - cutNum - 1 - self.fs)
                    absolute_position, relative_position = self.getNextMaxValuePositon(sig, 300, randomPosition)
                    sig_segment = sig[absolute_position:absolute_position + cutNum]
                    logFbank = self.melSpectrogram(sig_segment, 300)
                    myData.append(logFbank)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 2:
                for j in range(8 * mutiplesNumber):  # 2
                    randomPosition = random.randint(0, sigLen - cutNum - 1 - self.fs)
                    absolute_position, relative_position = self.getNextMaxValuePositon(sig, 300, randomPosition)
                    sig_segment = sig[absolute_position:absolute_position + cutNum]
                    logFbank = self.melSpectrogram(sig_segment, 300)
                    myData.append(logFbank)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 3:
                for j in range(70 * mutiplesNumber):  # 120
                    randomPosition = random.randint(0, sigLen - cutNum - 1 - self.fs)
                    absolute_position, relative_position = self.getNextMaxValuePositon(sig, 300, randomPosition)
                    sig_segment = sig[absolute_position:absolute_position + cutNum]
                    logFbank = self.melSpectrogram(sig_segment, 300)
                    myData.append(logFbank)
                    myLabel.append(AllLabel[fileName])
                    print(logFbank.shape)
        myData = np.asarray(myData)
        myLabel = np.asarray(myLabel)
        return myData, myLabel

    # from a siganl ,get position list of max value
    def getMaxPostionList(self, signal, fs=300):
        envmax = signal
        position_recorder = []
        max_first_position = np.argmax(envmax[0:fs])
        iterator = max_first_position
        t_envmax = envmax.T
        # find max value positon###########
        while iterator < len(envmax):  # find from the first pitch
            position_recorder.append(iterator)
            if len(envmax) > iterator + fs:
                start_pos = iterator + fs / 4
                end_pos = iterator + fs
            else:
                break
            max_local_position = np.argmax(t_envmax[start_pos:end_pos])
            iterator = max_local_position + start_pos
        return position_recorder

    def randomGetNextMaxValuePositon(self, signal, fs, cutLength):
        randomPositon = random.randint(0, len(
            signal) - cutLength - self.fs - 1)  # random choose a position in time in signal

        absolute_position = randomPositon + np.argmax(signal[randomPositon:randomPositon + fs])

        return absolute_position


    #Segemnt signal , get raw data
    def RandomCutSingleRaw(self, AllSig, AllLabel, fileNameList, cutNum=4 * 300,
                           mutiplesNumber=1):  # 每个mat文件只取一个数据，数据量少的多取，目的是平衡数据
        myData = []
        myLabel = []
        for fileName in fileNameList:
            sig = AllSig[fileName]
            sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
            #
            f = FIRFilters()
            filtecg = f.highpass(sig, 300, 1)
            filtecg = f.lowpass(filtecg, 300, 70)
            #
            sig = filtecg[self.fs:]  # first second is not reliable
            sigLen = len(sig)
            print(fileName)
            if AllLabel[fileName] == 0:
                for j in range(4 * mutiplesNumber):  # 1
                    randomPosition = random.randint(0,
                                                    sigLen - cutNum - 1 - self.fs)  # random choose a position in time in signal
                    absolute_position, relative_position = self.getNextMaxValuePositon(sig, 300, randomPosition)
                    sig_segment = sig[absolute_position:absolute_position + cutNum]

                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 1:
                for j in range(27 * mutiplesNumber):  # 7
                    randomPosition = random.randint(0, sigLen - cutNum - 1 - self.fs)
                    absolute_position, relative_position = self.getNextMaxValuePositon(sig, 300, randomPosition)
                    sig_segment = sig[absolute_position:absolute_position + cutNum]

                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 2:
                for j in range(8 * mutiplesNumber):  # 2
                    randomPosition = random.randint(0, sigLen - cutNum - 1 - self.fs)
                    absolute_position, relative_position = self.getNextMaxValuePositon(sig, 300, randomPosition)
                    sig_segment = sig[absolute_position:absolute_position + cutNum]
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 3:
                for j in range(70 * mutiplesNumber):  # 120
                    randomPosition = random.randint(0, sigLen - cutNum - 1 - self.fs)
                    absolute_position, relative_position = self.getNextMaxValuePositon(sig, 300, randomPosition)
                    sig_segment = sig[absolute_position:absolute_position + cutNum]

                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
                    print(sig_segment.shape)
        myData = np.asarray(myData)
        myLabel = np.asarray(myLabel)
        return myData, myLabel
        # Read lobfbankData ,2D data     # Load data and shaffle data

    def RandomCutSingleRaw2(self, AllSig, AllLabel, fileNameList, cutLength=4 * 300,
                            mutiplesNumber=1):  # 每个mat文件只取一个数据，数据量少的多取，目的是平衡数据
        myData = []
        myLabel = []
        for fileName in fileNameList:
            sig = AllSig[fileName]
            sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
            #
            f = FIRFilters()
            filtecg = f.highpass(sig, 300, 1)
            filtecg = f.lowpass(filtecg, 300, 70)
            #
            sig = filtecg[self.fs:]  # first second is not reliable
            sigLen = len(sig)
            print(fileName)
            if AllLabel[fileName] == 0:
                for j in range(4 * mutiplesNumber):  # 1
                    randomPosition = random.randint(0,
                                                    sigLen - cutLength - 1)  # random choose a position in time in signal
                    sig_segment = sig[randomPosition:randomPosition + cutLength]
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 1:
                for j in range(27 * mutiplesNumber):  # 7
                    randomPosition = random.randint(0,
                                                    sigLen - cutLength - 1)  # random choose a position in time in signal
                    sig_segment = sig[randomPosition:randomPosition + cutLength]
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 2:
                for j in range(8 * mutiplesNumber):  # 2
                    randomPosition = random.randint(0,
                                                    sigLen - cutLength - 1)  # random choose a position in time in signal
                    sig_segment = sig[randomPosition:randomPosition + cutLength]
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 3:
                for j in range(70 * mutiplesNumber):  # 120
                    randomPosition = random.randint(0,
                                                    sigLen - cutLength - 1)  # random choose a position in time in signal
                    sig_segment = sig[randomPosition:randomPosition + cutLength]
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
                    print(sig_segment.shape)
        myData = np.asarray(myData)
        myLabel = np.asarray(myLabel)
        return myData, myLabel
    def readFbankData(self, img_rows=39, img_cols=40, outputClass=4):
        file = h5py.File('TrainTestData.h5', 'r')
        trainData = file['TrainList']
        trainLabel = file['TrainLabelList']
        testData = file['TestList']
        testLable = file['TestLabelList']
        numTrain = len(trainData)

        def PreProcessing(data, dataLabel):
            print(np.asarray(data).shape)
            data = np.asarray(data).reshape((-1, img_rows, img_cols))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], 1))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable

        print("Load from hdf5 file,the shap is ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)
        testData, testLable = PreProcessing(testData, testLable)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
        testData, testLable = shuffle(testData, testLable, random_state=0)
        # generate valide data
        numSplit = int(round(0.8 * len(testData)))
        X_test = testData[:numSplit]
        y_test = testLable[:numSplit]
        X_valid = testData[numSplit:]
        y_valid = testLable[numSplit:]

        X_train = trainData
        y_train = trainLabel

        # reshape data
        X_train = np.reshape(X_train, (-1, 1, img_rows, img_cols))
        X_valid = np.reshape(X_valid, (-1, 1, img_rows, img_cols))
        X_test = np.reshape(X_test, (-1, 1, img_rows, img_cols))

        Y_train = to_categorical(y_train, outputClass)
        Y_valid = to_categorical(y_valid, outputClass)
        Y_test = to_categorical(y_test, outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        file.close()
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

    def get_arguments(self, nb_epoch, batch_size, SaveName, IsLoadWeights, LoadWeightsName):
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
        parser.add_argument('--SaveName', type=str, default=SaveName,
                            help='Save weight Name')
        parser.add_argument('--Train', type=_str_to_bool, default=True,
                            help='Train? if Train=False,the program will not train ,will load savePreNme.weight ,and evaluate the test ')
        parser.add_argument('--IsLoadWeights', type=_str_to_bool, default=IsLoadWeights,
                            help='Train? if Train=False,the program will not train ,will load savePreNme.weight ,and evaluate the test ')
        parser.add_argument('--LoadWeightsName', type=str, default=LoadWeightsName,
                            help='SavePreName')

        return parser.parse_args()

    def GetRawSegmentArray(self, sig, fs=300, segLength=300, segNum=40):
        randomSegment_Array = []
        maxValuePositionList = self.getMaxPostionList(sig, fs)  # get max value positionlist
        for k in range(segNum):
            randomPosition = random.randint(0, len(maxValuePositionList) - 2)
            start_pos = maxValuePositionList[randomPosition]
            end_pos = maxValuePositionList[randomPosition + 1]
            sig_segment = sig[start_pos:end_pos]
            sigLeng = len(sig_segment)
            if (sigLeng < segLength):
                addLength = segLength - sigLeng
                add_seg = np.zeros((1, addLength))

                sig_segment = sig_segment.reshape((1, -1))
                print("addLength, sigLeng", addLength, sigLeng)
                print("add_seg.shape, sig_segment.shape", add_seg.shape, sig_segment.shape)
                sig_segment = np.hstack((sig_segment, add_seg))
            else:
                sig_segment = sig_segment[0:segLength]

            randomSegment_Array.append(sig_segment)
        randomSegment_Array = np.asarray(randomSegment_Array)  # get 40*400 array
        return randomSegment_Array

    # no padding zero
    def GetRawSegmentArrayNoPadding(self, sig, fs=300, segLength=300, segNum=20):
        randomSegment_Array = []
        sig_cutLastsection = sig[:len(sig) - segLength]  # avoid segment exceed sig max length
        maxValuePositionList = self.getMaxPostionList(sig_cutLastsection, fs)  # get max value positionlist
        for k in range(segNum):
            randomPosition = random.randint(0, len(maxValuePositionList) - 1)
            start_pos = maxValuePositionList[randomPosition]
            end_pos = start_pos + segLength
            sig_segment = sig[start_pos:end_pos]
            randomSegment_Array.append(sig_segment)
        randomSegment_Array = np.asarray(randomSegment_Array)  # get 40*400 array
        return randomSegment_Array

    # no padding zero
    def GetRawSegmentArrayNoPadding_Random(self, sig, fs=300, segLength=300, segNum=20):
        randomSegment_Array = []
        sig_cutLastsection = len(sig) - segLength  # avoid segment exceed sig max length
        for k in range(segNum):
            randomPosition = random.randint(0, sig_cutLastsection - 1)
            start_pos = randomPosition
            end_pos = start_pos + segLength
            sig_segment = sig[start_pos:end_pos]
            randomSegment_Array.append(sig_segment)
        randomSegment_Array = np.asarray(randomSegment_Array)  # get 40*400 array
        return randomSegment_Array
    # generate raw array data
    def getRawArrayData(self, AllSig, AllLabel, fileNameList, cutLength=1 * 300,
                        mutiplesNumber=1):  # 每个mat文件只取一个数据，数据量少的多取，目的是平衡数据
        myData = []
        myLabel = []
        for fileName in fileNameList:
            sig = AllSig[fileName]
            sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
            #
            f = FIRFilters()
            filtecg = f.highpass(sig, 300, 1)
            filtecg = f.lowpass(filtecg, 300, 70)
            #
            sig = filtecg[self.fs:]  # first second is not reliable
            sigLen = len(sig)
            print(fileName)
            if AllLabel[fileName] == 0:
                for j in range(4 * mutiplesNumber):  # 1
                    sig_segment = self.GetRawSegmentArrayNoPadding_Random(sig, self.fs, cutLength, segNum=3)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 1:
                for j in range(27 * mutiplesNumber):  # 7
                    sig_segment = self.GetRawSegmentArrayNoPadding_Random(sig, self.fs, cutLength, segNum=3)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 2:
                for j in range(8 * mutiplesNumber):  # 2
                    sig_segment = self.GetRawSegmentArrayNoPadding_Random(sig, self.fs, cutLength, segNum=3)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 3:
                for j in range(70 * mutiplesNumber):  # 120
                    sig_segment = self.GetRawSegmentArrayNoPadding_Random(sig, self.fs, cutLength, segNum=3)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
                    print(sig_segment.shape)
        myData = np.asarray(myData)
        myLabel = np.asarray(myLabel)
        return myData, myLabel

    # using in getRawFilterData
    def filterFun(self, sig, fs=300, lowPass=1, highPass=150):
        f = FIRFilters()
        filtecg = f.highpass(sig, fs, lowPass)
        filtecg = f.lowpass(filtecg, fs, highPass)
        return filtecg

    def GetRawFilterSegmentArray(self, sig, fs=300, segLength=300, segNum=20):
        randomSegment_Array = []
        sig_cutLastsection = len(sig) - segLength  # avoid segment exceed sig max length
        for k in range(segNum):
            randomPosition = random.randint(0, sig_cutLastsection - 1)
            start_pos = randomPosition
            end_pos = start_pos + segLength
            sig_segment = sig[start_pos:end_pos]
            sig_filter1 = self.filterFun(sig_segment, self.fs, 1, 70)
            sig_filter2 = self.filterFun(sig_segment, self.fs, 5, 90)
            sig_filter3 = self.filterFun(sig_segment, self.fs, 1, 150)
            randomSegment_Array.append(sig_filter1)
            randomSegment_Array.append(sig_filter2)
            randomSegment_Array.append(sig_filter3)
        randomSegment_Array = np.asarray(randomSegment_Array)  # get 40*400 array
        return randomSegment_Array

    # filter array
    def getRawFilterData(self, AllSig, AllLabel, fileNameList, cutLength=1 * 300,
                         mutiplesNumber=1):  # 每个mat文件只取一个数据，数据量少的多取，目的是平衡数据
        myData = []
        myLabel = []
        for fileName in fileNameList:
            sig = AllSig[fileName]
            sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
            sig = sig[self.fs:]  # first second is not reliable
            print(fileName)
            if AllLabel[fileName] == 0:
                for j in range(4 * mutiplesNumber):  # 1
                    sig_segment = self.GetRawFilterSegmentArray(sig, self.fs, cutLength, segNum=3)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 1:
                for j in range(27 * mutiplesNumber):  # 7
                    sig_segment = self.GetRawFilterSegmentArray(sig, self.fs, cutLength, segNum=3)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 2:
                for j in range(8 * mutiplesNumber):  # 2
                    sig_segment = self.GetRawFilterSegmentArray(sig, self.fs, cutLength, segNum=3)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 3:
                for j in range(70 * mutiplesNumber):  # 120
                    sig_segment = self.GetRawFilterSegmentArray(sig, self.fs, cutLength, segNum=3)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
                    print(sig_segment.shape)
        myData = np.asarray(myData)
        myLabel = np.asarray(myLabel)
        return myData, myLabel

    # connect different segment,not append
    def GetRawFilterSegmentArray2(self, sig, fs=300, segLength=300, segNum=3):
        randomSegment_Array = []
        segmentAll = []
        sig_cutLastsection = len(sig) - segLength  # avoid segment exceed sig max length
        for k in range(segNum):
            randomPosition = random.randint(0, sig_cutLastsection - 1)
            start_pos = randomPosition
            end_pos = start_pos + segLength
            sig_segment = sig[start_pos:end_pos]
            segmentAll[len(segmentAll):len(sig_segment)] = sig_segment

        segmentAll = np.asarray(segmentAll)
        # print("segmentAll",segmentAll.shape)

        sig_filter1 = self.filterFun(segmentAll, self.fs, 1, 70)
        sig_filter2 = self.filterFun(segmentAll, self.fs, 5, 90)
        sig_filter3 = self.filterFun(segmentAll, self.fs, 20, 120)
        randomSegment_Array.append(segmentAll)
        randomSegment_Array.append(sig_filter1)
        randomSegment_Array.append(sig_filter2)
        randomSegment_Array.append(sig_filter3)
        randomSegment_Array = np.asarray(randomSegment_Array)  # get 4*6s array
        # print(randomSegment_Array.shape)
        return randomSegment_Array

    # connect different segment,not append
    # return absolute position and relative position
    def getNextMaxValuePositon(self, signal, fs, randomPositon):
        absolute_position = randomPositon + np.argmax(signal[randomPositon:randomPositon + fs])
        relative_position = np.argmax(signal[randomPositon:randomPositon + fs])
        return absolute_position, relative_position

    #
    def getStartNextMaxValuePositon(self, signal, fs, startPosition):
        absolute_position = startPosition + np.argmax(signal[startPosition:startPosition + fs])
        # relative_position = np.argmax(signal[startPosition:startPosition + fs])
        return absolute_position

    def getEndLastMaxValuePositon(self, signal, fs, endPosition):
        absolute_position = endPosition + np.argmax(signal[endPosition:endPosition + fs])
        # relative_position = np.argmax(signal[endPosition:endPosition + fs])
        return absolute_position

    def GetRawFilterSegmentArray_MaxPointJoint(self, sig, fs=300, segLength=3 * 300, segNum=4):
        randomSegment_Array = []
        segmentAll = []

        sig_cutLastsection = len(sig) - segLength - fs  # avoid segment exceed sig max length

        for k in range(segNum):
            randomPosition = random.randint(0, sig_cutLastsection - 1)
            start_pos = randomPosition
            end_pos = start_pos + segLength + fs
            if k == 0:
                start_pos = start_pos  # first start position don't need change
            else:
                start_pos = self.getStartNextMaxValuePositon(sig, self.fs, start_pos)
            if k == segNum - 1:
                end_pos = end_pos  # last end position extend 1 second ,For simplifying caculation
            else:
                end_pos = self.getEndLastMaxValuePositon(sig, self.fs, end_pos)
            sig_segment = sig[start_pos:end_pos]
            segmentAll[len(segmentAll):len(sig_segment)] = sig_segment
        if (len(segmentAll) > segLength * segNum):
            segmentAll = segmentAll[:segLength * segNum]
        segmentAll = np.asarray(segmentAll)
        sig_filter1 = self.filterFun(segmentAll, self.fs, 1, 70)
        sig_filter2 = self.filterFun(segmentAll, self.fs, 5, 90)
        sig_filter3 = self.filterFun(segmentAll, self.fs, 20, 120)
        randomSegment_Array.append(segmentAll)
        randomSegment_Array.append(sig_filter1)
        randomSegment_Array.append(sig_filter2)
        randomSegment_Array.append(sig_filter3)
        randomSegment_Array = np.asarray(randomSegment_Array)  # get 4*6s array
        #print(randomSegment_Array.shape)
        return randomSegment_Array
    def getRawFilterArray(self, AllSig, AllLabel, fileNameList, cutLength=1 * 300,
                          mutiplesNumber=1):  # 每个mat文件只取一个数据，数据量少的多取，目的是平衡数据
        myData = []
        myLabel = []
        for fileName in fileNameList:
            sig = AllSig[fileName]
            sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
            sig = sig[self.fs:]  # first second is not reliable
            #print(fileName)
            if AllLabel[fileName] == 0:
                for j in range(4 * mutiplesNumber):  # 1
                    sig_segment = self.GetRawFilterSegmentArray_MaxPointJoint(sig, self.fs, cutLength, segNum=4)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 1:
                for j in range(27 * mutiplesNumber):  # 7
                    sig_segment = self.GetRawFilterSegmentArray_MaxPointJoint(sig, self.fs, cutLength, segNum=4)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 2:
                for j in range(8 * mutiplesNumber):  # 2
                    sig_segment = self.GetRawFilterSegmentArray_MaxPointJoint(sig, self.fs, cutLength, segNum=4)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 3:
                for j in range(70 * mutiplesNumber):  # 120
                    sig_segment = self.GetRawFilterSegmentArray_MaxPointJoint(sig, self.fs, cutLength, segNum=4)
                    myData.append(sig_segment)
                    myLabel.append(AllLabel[fileName])
                    #print(sig_segment.shape)
        myData = np.asarray(myData)
        myLabel = np.asarray(myLabel)
        return myData, myLabel
    def fromRawArrayToFbank(self, rawArray, fs=300):
        fbankArray = []
        for rawSegment in rawArray:
            logFbank = self.melSpectrogram(rawSegment, self.fs)
            #print(logFbank.shape)
            fbankArray.append(logFbank)
        fbankArray = np.asarray(fbankArray)
        return fbankArray

    def getFbankArrayData(self, AllSig, AllLabel, fileNameList, cutLength=4 * 300,
                          mutiplesNumber=1):  # 每个mat文件只取一个数据，数据量少的多取，目的是平衡数据
        myData = []
        myLabel = []
        for fileName in fileNameList:
            sig = AllSig[fileName]
            sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
            #add _filter
            f = FIRFilters()
            filtecg = f.highpass(sig, 300, 2)
            filtecg = f.lowpass(filtecg, 300, 120)
            ###################
            sig = filtecg[self.fs:]  # first second is not reliable
            sigLen = len(sig)
            print(fileName)
            if AllLabel[fileName] == 0:
                for j in range(4 * mutiplesNumber):  # 1 the number of geting segment from a signal
                    rawSegmentArray = self.GetRawSegmentArrayNoPadding(sig, self.fs, cutLength, 7)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank(
                        rawSegmentArray)  # get 10*39*40 number * step * specvector
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 1:
                for j in range(27 * mutiplesNumber):  # 7
                    rawSegmentArray = self.GetRawSegmentArrayNoPadding(sig, self.fs, cutLength, 7)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank(
                        rawSegmentArray)  # get 10*39*40 number * step * specvector
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 2:
                for j in range(8 * mutiplesNumber):  # 2
                    rawSegmentArray = self.GetRawSegmentArrayNoPadding(sig, self.fs, cutLength, 7)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank(
                        rawSegmentArray)  # get 10*39*40 number * step * specvector
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 3:
                for j in range(70 * mutiplesNumber):  # 120
                    rawSegmentArray = self.GetRawSegmentArrayNoPadding(sig, self.fs, cutLength, 7)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank(
                        rawSegmentArray)  # get 10*39*40 number * step * specvector
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])
                    # print(fbankSegmentArray.shape)
        myData = np.asarray(myData)
        myLabel = np.asarray(myLabel)
        return myData, myLabel

    def getFbankArrayData_Random(self, AllSig, AllLabel, fileNameList, cutLength=6 * 300,
                                 mutiplesNumber=1):  # 每个mat文件只取一个数据，数据量少的多取，目的是平衡数据
        myData = []
        myLabel = []
        for fileName in fileNameList:
            sig = AllSig[fileName]
            sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
            # add _filter
            f = FIRFilters()
            filtecg = f.highpass(sig, 300, 1)
            filtecg = f.lowpass(filtecg, 300, 120)
            ###################
            sig = filtecg[self.fs:]  # first second is not reliable
            sigLen = len(sig)
            print(fileName)
            if AllLabel[fileName] == 0:
                for j in range(4 * mutiplesNumber):  # 1 the number of geting segment from a signal
                    rawSegmentArray = self.GetRawSegmentArrayNoPadding_Random(sig, self.fs, cutLength, 7)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank(
                        rawSegmentArray)  # get 10*39*40 number * step * specvector
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 1:
                for j in range(27 * mutiplesNumber):  # 7
                    rawSegmentArray = self.GetRawSegmentArrayNoPadding_Random(sig, self.fs, cutLength, 7)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank(
                        rawSegmentArray)  # get 10*39*40 number * step * specvector
                    print(fbankSegmentArray.shape)
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 2:
                for j in range(8 * mutiplesNumber):  # 2
                    rawSegmentArray = self.GetRawSegmentArrayNoPadding_Random(sig, self.fs, cutLength, 7)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank(
                        rawSegmentArray)  # get 10*39*40 number * step * specvector
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 3:
                for j in range(70 * mutiplesNumber):  # 120
                    rawSegmentArray = self.GetRawSegmentArrayNoPadding_Random(sig, self.fs, cutLength, 7)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank(
                        rawSegmentArray)  # get 10*39*40 number * step * specvector
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])
                    # print(fbankSegmentArray.shape)
        myData = np.asarray(myData)
        myLabel = np.asarray(myLabel)
        return myData, myLabel
    # improved versiton 2:add 2s data##################################################
    def getFbankArrayData_2(self, AllSig, AllLabel, fileNameList, cutLength=4 * 300,
                            mutiplesNumber=1):  # 每个mat文件只取一个数据，数据量少的多取，目的是平衡数据
        myData = []
        myLabel = []
        for fileName in fileNameList:
            sig = AllSig[fileName]
            sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
            sig = sig[self.fs:]  # first second is not reliable

            print(fileName)
            if AllLabel[fileName] == 0:
                for j in range(4 * mutiplesNumber):  # 1 the number of geting segment from a signal
                    rawSegmentArray = self.GetRawSegmentArrayNoPadding2(sig, self.fs, cutLength, 7)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank2(
                        rawSegmentArray)  # get 10*39*40 number * step * specvector
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 1:
                for j in range(27 * mutiplesNumber):  # 7
                    rawSegmentArray = self.GetRawSegmentArrayNoPadding2(sig, self.fs, cutLength, 7)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank2(
                        rawSegmentArray)  # get 10*39*40 number * step * specvector
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 2:
                for j in range(8 * mutiplesNumber):  # 2
                    rawSegmentArray = self.GetRawSegmentArrayNoPadding2(sig, self.fs, cutLength, 7)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank2(
                        rawSegmentArray)  # get 10*39*40 number * step * specvector
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 3:
                for j in range(70 * mutiplesNumber):  # 120
                    fbankSegmentArray = self.fromRawArrayToFbank2(
                        rawSegmentArray)  # get 10*39*40 number * step * specvector
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])
                    # print(fbankSegmentArray.shape)
        myData = np.asarray(myData)
        myLabel = np.asarray(myLabel)
        return myData, myLabel
    # no padding zero
    def GetRawSegmentArrayNoPadding2(self, sig, fs=300, segLength=4 * 300, segNum=7):
        randomSegment_Array = []

        # 2s:2*2s
        for k in range(2):  # 2second is 4
            cutLength = 2 * fs
            start_pos = self.randomGetNextMaxValuePositon(sig, self.fs, cutLength)
            end_pos = start_pos + cutLength
            sig_segment = sig[start_pos:end_pos]
            randomSegment_Array.append(sig_segment)
        # 4s:2*4s
        for k in range(2):  # 4second is 3
            cutLength = 4 * fs
            start_pos = self.randomGetNextMaxValuePositon(sig, self.fs, cutLength)
            end_pos = start_pos + cutLength
            sig_segment = sig[start_pos:end_pos]
            randomSegment_Array.append(sig_segment)

        # 6s:2*6s
        for k in range(2):  #2second is 4
            cutLength = 6 * fs
            start_pos = self.randomGetNextMaxValuePositon(sig, self.fs, cutLength)
            end_pos = start_pos + cutLength
            sig_segment = sig[start_pos:end_pos]
            randomSegment_Array.append(sig_segment)
        # 7s
        cutLength = 7 * fs
        if (len(sig) < (cutLength + fs)):
            start_pos = 0
        else:
            start_pos = self.randomGetNextMaxValuePositon(sig, self.fs, cutLength)
        end_pos = start_pos + cutLength
        sig_segment = sig[start_pos:end_pos]
        randomSegment_Array.append(sig_segment)
        #print(len(randomSegment_Array))
        return randomSegment_Array
    def fromRawArrayToFbank2(self, rawArray, segNum=7):
        fbankArray = []
        for rawSegment in rawArray[0:2]:
            logFbank = self.melSpectrogram(rawSegment, self.fs, winlen=0.1, winstep=0.05)  #print(logFbank.shape)
            print(logFbank.shape)
            fbankArray.append(logFbank)
        for rawSegment in rawArray[2:4]:
            logFbank = self.melSpectrogram(rawSegment, self.fs, winlen=0.2, winstep=0.1)
            print(logFbank.shape)
            fbankArray.append(logFbank)
        for rawSegment in rawArray[4:6]:
            logFbank = self.melSpectrogram(rawSegment, self.fs, winlen=0.3, winstep=0.15)
            print(logFbank.shape)
            fbankArray.append(logFbank)
        rawSegment = rawArray[6]
        logFbank = self.melSpectrogram(rawSegment, self.fs, winlen=0.35, winstep=0.175)  # for 2s
        print(logFbank.shape)
        fbankArray.append(logFbank)
        fbankArray = np.asarray(fbankArray)
        if (fbankArray.shape[1] != 39):
            print(logFbank.shape)
            for i in rawArray:
                print(len(rawArray[i]))
        print(fbankArray.shape)
        return fbankArray

    def getMixData(self, AllSig, AllLabel, fileNameList, cutLength=4 * 300,
                   mutiplesNumber=1):  # 每个mat文件只取一个数据，数据量少的多取，目的是平衡数据
        myData = []
        myRawData =[]
        myLabel = []
        for fileName in fileNameList:
            sig = AllSig[fileName]
            sig = sig.reshape([1, -1]).flatten()  # 9000*1 tranform to 1*9000
            sig = sig[self.fs:]  # first second is not reliable
            sigLen = len(sig)
            print(fileName)
            if AllLabel[fileName] == 0:
                for j in range(4 * mutiplesNumber):  # 1 the number of geting segment from a signal
                    rawSegmentArray = self.GetRawSegmentArrayNoPadding(sig, self.fs, cutLength, 20)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank(
                        rawSegmentArray[0:7])  # get 10*39*40 number * step * specvector
                    myRawData.append(rawSegmentArray)
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])
            elif AllLabel[fileName] == 1:
                for j in range(27 * mutiplesNumber):  # 7
                    rawSegmentArray = self.GetRawSegmentArrayNoPadding(sig, self.fs, cutLength, 20)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank(
                        rawSegmentArray[0:7])  # get 10*39*40 number * step * specvector
                    myRawData.append(rawSegmentArray)
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 2:
                for j in range(8 * mutiplesNumber):  # 2
                    rawSegmentArray = self.GetRawSegmentArrayNoPadding(sig, self.fs, cutLength, 20)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank(
                        rawSegmentArray[0:7])  # get 10*39*40 number * step * specvector
                    myRawData.append(rawSegmentArray)
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])

            elif AllLabel[fileName] == 3:
                for j in range(70 * mutiplesNumber):  # 120
                    rawSegmentArray = self.GetRawSegmentArrayNoPadding(sig, self.fs, cutLength, 20)  # get 40*300
                    fbankSegmentArray = self.fromRawArrayToFbank(
                        rawSegmentArray[0:7])  # get 10*39*40 number * step * specvector
                    myRawData.append(rawSegmentArray)
                    myData.append(fbankSegmentArray)
                    myLabel.append(AllLabel[fileName])
                    # print(fbankSegmentArray.shape)
        myRawData = np.asarray(myRawData)
        myData = np.asarray(myData)
        myLabel = np.asarray(myLabel)
        return myData, myLabel, myRawData
    ###########################################################################
    def readArrayData(self, img_rows=40, img_cols=300, outputClass=4):
        file = h5py.File('TrainTestDataArray.h5', 'r')
        trainData = file['TrainList']
        trainLabel = file['TrainLabelList']
        testData = file['TestList']
        testLable = file['TestLabelList']
        numTrain = len(trainData)

        def PreProcessing(data, dataLabel):
            print(np.asarray(data).shape)
            data = np.asarray(data).reshape((-1, img_rows, img_cols))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], 1))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable

        print("Load from hdf5 file,the shap is ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)
        testData, testLable = PreProcessing(testData, testLable)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
        testData, testLable = shuffle(testData, testLable, random_state=0)
        # generate valide data
        numSplit = int(round(0.8 * len(testData)))
        X_test = testData[:numSplit]
        y_test = testLable[:numSplit]
        X_valid = testData[numSplit:]
        y_valid = testLable[numSplit:]

        X_train = trainData
        y_train = trainLabel

        # reshape data
        X_train = np.reshape(X_train, (-1, 1, img_rows, img_cols))
        X_valid = np.reshape(X_valid, (-1, 1, img_rows, img_cols))
        X_test = np.reshape(X_test, (-1, 1, img_rows, img_cols))

        Y_train = to_categorical(y_train, outputClass)
        Y_valid = to_categorical(y_valid, outputClass)
        Y_test = to_categorical(y_test, outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        file.close()
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

    def readFbankArrayData(self, inputNum=7, img_rows=39, img_cols=40, outputClass=4,
                           fileName='TrainTestDataFbankArray.h5'):
        file = h5py.File(fileName, 'r')
        trainData = file['TrainList']
        trainLabel = file['TrainLabelList']
        testData = file['TestList']
        testLable = file['TestLabelList']
        numTrain = len(trainData)

        def PreProcessing(data, dataLabel):
            print(np.asarray(data).shape)
            data = np.asarray(data).reshape((-1, inputNum, img_rows, img_cols))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], 1))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable

        print("Load from hdf5 file,the shap is ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)
        testData, testLable = PreProcessing(testData, testLable)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
        testData, testLable = shuffle(testData, testLable, random_state=0)
        # generate valide data
        numSplit = int(round(0.8 * len(testData)))
        X_test = testData[:numSplit]
        y_test = testLable[:numSplit]
        X_valid = testData[numSplit:]
        y_valid = testLable[numSplit:]

        X_train = trainData
        y_train = trainLabel

        # reshape data
        X_train = np.reshape(X_train, (-1, inputNum, img_rows, img_cols))
        X_valid = np.reshape(X_valid, (-1, inputNum, img_rows, img_cols))
        X_test = np.reshape(X_test, (-1, inputNum, img_rows, img_cols))

        Y_train = to_categorical(y_train, outputClass)
        Y_valid = to_categorical(y_valid, outputClass)
        Y_test = to_categorical(y_test, outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        file.close()
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

    def processMulInput(self, X_input, img_rows=39, img_cols=40):
        X_input = np.swapaxes(X_input, 0, 1)
        X_input0 = X_input[0]
        X_input1 = X_input[1]
        X_input2 = X_input[2]
        X_input3 = X_input[3]
        X_input4 = X_input[4]
        X_input5 = X_input[5]
        X_input6 = X_input[6]
        X_input0 = X_input0.reshape(-1, 1, img_rows, img_cols)
        X_input1 = X_input1.reshape(-1, 1, img_rows, img_cols)
        X_input2 = X_input2.reshape(-1, 1, img_rows, img_cols)
        X_input3 = X_input3.reshape(-1, 1, img_rows, img_cols)
        X_input4 = X_input4.reshape(-1, 1, img_rows, img_cols)
        X_input5 = X_input5.reshape(-1, 1, img_rows, img_cols)
        X_input6 = X_input6.reshape(-1, 1, img_rows, img_cols)

        return X_input0, X_input1, X_input2, X_input3, X_input4, X_input5, X_input6

    def processMulRawInput(self, X_input, inputTickdim=6 * 300, inputFeatureDim=1):
        X_input = np.swapaxes(X_input, 0, 1)
        X_input0 = X_input[0]
        X_input1 = X_input[1]
        X_input2 = X_input[2]
        X_input0 = X_input0.reshape(-1, inputTickdim, inputFeatureDim)
        X_input1 = X_input1.reshape(-1, inputTickdim, inputFeatureDim)
        X_input2 = X_input2.reshape(-1, inputTickdim, inputFeatureDim)
        return X_input0, X_input1, X_input2

    def savePredictAndTarget(self, y_test, y_pred, saveName=""):
        print("predict shape", y_pred.shape)
        directoryName = 'PredictResult'
        print("save feature Data")

        print("save predictLabel Data")
        np.save(directoryName + '/' + saveName + "predict.npy", np.asarray(y_pred))
        print("save targetLabel Data")
        np.save(directoryName + '/' + saveName + "target.npy", np.asarray(y_test))

    def readRawData(self, inputNum=1, inputTickdim=1200, inputFeatureDim=1, outputClass=4,
                    fileName='TrainTestRaw.h5'):
        file = h5py.File(fileName, 'r')
        trainData = file['TrainList']
        trainLabel = file['TrainLabelList']
        testData = file['TestList']
        testLable = file['TestLabelList']
        numTrain = len(trainData)

        def PreProcessing(data, dataLabel):
            print(np.asarray(data).shape)
            numData = len(data)
            data = np.asarray(data).reshape((numData, inputTickdim * inputFeatureDim))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], 1))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable

        print("Load from hdf5 file,the shap is ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)
        testData, testLable = PreProcessing(testData, testLable)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
        testData, testLable = shuffle(testData, testLable, random_state=0)
        # generate valide data
        numSplit = int(round(0.8 * len(testData)))
        X_test = testData[:numSplit]
        y_test = testLable[:numSplit]
        X_valid = testData[numSplit:]
        y_valid = testLable[numSplit:]

        X_train = trainData
        y_train = trainLabel

        # reshape data
        X_train = np.reshape(X_train, (-1, inputTickdim, inputFeatureDim))
        X_valid = np.reshape(X_valid, (-1, inputTickdim, inputFeatureDim))
        X_test = np.reshape(X_test, (-1, inputTickdim, inputFeatureDim))

        Y_train = to_categorical(y_train, outputClass)
        Y_valid = to_categorical(y_valid, outputClass)
        Y_test = to_categorical(y_test, outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        file.close()
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

    def readRawArrayData(self, inputNum=1, inputTickdim=1200, inputFeatureDim=20, outputClass=4,
                         fileName='TrainTestRaw.h5'):
        file = h5py.File(fileName, 'r')
        trainData = file['TrainList']
        trainLabel = file['TrainLabelList']
        testData = file['TestList']
        testLable = file['TestLabelList']
        numTrain = len(trainData)

        def PreProcessing(data, dataLabel):
            print(np.asarray(data).shape)
            numData = len(data)
            data = np.swapaxes(data, 1, 2)
            data = np.asarray(data).reshape((numData, inputTickdim, inputFeatureDim))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], 1))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable

        print("Load from hdf5 file,the shap is ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)
        testData, testLable = PreProcessing(testData, testLable)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
        testData, testLable = shuffle(testData, testLable, random_state=0)
        # generate valide data
        numSplit = int(round(0.8 * len(testData)))
        # X_train = trainData[:numSplit]
        # y_train = trainLabel[:numSplit]
        # X_valid = trainData[numSplit:numTrain]
        # y_valid = trainLabel[numSplit:numTrain]
        #
        # X_test = testData
        # y_test = testLable
        X_test = testData[:numSplit]
        y_test = testLable[:numSplit]
        X_valid = testData[numSplit:]
        y_valid = testLable[numSplit:]

        X_train = trainData
        y_train = trainLabel

        # reshape data
        X_train = np.reshape(X_train, (-1, inputTickdim, inputFeatureDim))
        X_valid = np.reshape(X_valid, (-1, inputTickdim, inputFeatureDim))
        X_test = np.reshape(X_test, (-1, inputTickdim, inputFeatureDim))

        Y_train = to_categorical(y_train, outputClass)
        Y_valid = to_categorical(y_valid, outputClass)
        Y_test = to_categorical(y_test, outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        file.close()
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

    def prepareRawFilterArrayData(self, trainData, trainLabel, testData, testLabel, inputTickdim=1200,
                                  inputFeatureDim=20, outputClass=4):
        numTrain = len(trainData)

        def PreProcessing(data, dataLabel):
            print(np.asarray(data).shape)
            numData = len(data)
            data = np.swapaxes(data, 1, 2)
            data = np.asarray(data).reshape((numData, inputTickdim, inputFeatureDim))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], 1))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable

        print("prepare Data ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)
        testData, testLable = PreProcessing(testData, testLabel)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
        testData, testLable = shuffle(testData, testLable, random_state=0)
        # generate valide data
        numSplit = int(round(1 * len(testData)))

        X_test = testData[:numSplit]
        y_test = testLable[:numSplit]
        X_valid = testData[numSplit:]
        y_valid = testLable[numSplit:]

        X_train = trainData
        y_train = trainLabel

        # reshape data
        X_train = np.reshape(X_train, (-1, inputTickdim, inputFeatureDim))
        X_valid = np.reshape(X_valid, (-1, inputTickdim, inputFeatureDim))
        X_test = np.reshape(X_test, (-1, inputTickdim, inputFeatureDim))

        Y_train = to_categorical(y_train, outputClass)
        Y_valid = to_categorical(y_valid, outputClass)
        Y_test = to_categorical(y_test, outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
    def readMergeRawData(self, inputNum=3, inputTickdim=1200, inputFeatureDim=1, outputClass=4,
                         fileName='TrainTestRaw.h5'):
        file = h5py.File(fileName, 'r')
        trainData = file['TrainList']
        trainLabel = file['TrainLabelList']
        testData = file['TestList']
        testLable = file['TestLabelList']
        numTrain = len(trainData)

        def PreProcessing(data, dataLabel):
            print(np.asarray(data).shape)
            numData = len(data)
            data = np.asarray(data).reshape((numData, inputNum, inputTickdim * inputFeatureDim))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], 1))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable

        print("Load from hdf5 file,the shap is ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)
        testData, testLable = PreProcessing(testData, testLable)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)
        testData, testLable = shuffle(testData, testLable, random_state=0)
        # generate valide data
        numSplit = int(round(0.8 * len(testData)))
        X_test = testData[:numSplit]
        y_test = testLable[:numSplit]
        X_valid = testData[numSplit:]
        y_valid = testLable[numSplit:]

        X_train = trainData
        y_train = trainLabel

        # reshape data
        X_train = np.reshape(X_train, (-1, inputNum, inputTickdim, inputFeatureDim))
        X_valid = np.reshape(X_valid, (-1, inputNum, inputTickdim, inputFeatureDim))
        X_test = np.reshape(X_test, (-1, inputNum, inputTickdim, inputFeatureDim))

        Y_train = to_categorical(y_train, outputClass)
        Y_valid = to_categorical(y_valid, outputClass)
        Y_test = to_categorical(y_test, outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        file.close()
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
    def readMixData(self, inputNum=7, img_rows=39, img_cols=40, inputTickdim=1200, inputFeatureDim=20, outputClass=4,
                    fileName='TrainTestMix.h5'):
        file = h5py.File(fileName, 'r')
        rawTrainData = file['RawTrainList']
        rawTestData = file['RawTestList']
        trainData = file['TrainList']
        testData = file['TestList']
        trainLabel = file['TrainLabelList']
        testLable = file['TestLabelList']
        numTrain = len(trainData)

        def rawPreProcessing(data, dataLabel):
            print(np.asarray(data).shape)
            numData = len(data)
            data = np.swapaxes(data, 1, 2)
            data = np.asarray(data).reshape((numData, inputTickdim * inputFeatureDim))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], 1))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable
        def fbankPreProcessing(data, dataLabel):
            print(np.asarray(data).shape)
            data = np.asarray(data).reshape((data.shape[0], -1))
            lable = np.asarray(dataLabel, dtype=int).reshape((data.shape[0], 1))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable
        print("Load from hdf5 file,the shap is ")
        rawTrainData, trainLabel = rawPreProcessing(rawTrainData, trainLabel)
        rawTestData, testLable = rawPreProcessing(rawTestData, testLable)
        trainData, trainLabel = fbankPreProcessing(trainData, trainLabel)
        testData, testLable = fbankPreProcessing(testData, testLable)
        # shuffle the data
        all_temp = np.hstack((trainData, rawTrainData))
        all = np.hstack((all_temp, trainLabel))
        np.random.shuffle(all)  # note:use random.shuffle()is error,must add the term np

        trainData = all[:, :inputNum * img_rows * img_cols]
        rawTrainData = all[:, inputNum * img_rows * img_cols:-1]
        trainLabel = all[:, -1]
        #trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)

        # generate valide data
        numSplit = int(round(0.7 * len(testData)))
        X_train1 = trainData[:numSplit]
        y_train = trainLabel[:numSplit]
        X_train2 = rawTrainData[:numSplit]


        X_valid1 = trainData[numSplit:numTrain]
        X_valid2 = rawTrainData[numSplit:numTrain]
        y_valid = trainLabel[numSplit:numTrain]

        X_test1 = testData
        X_test2 = rawTestData
        y_test = testLable

        # reshape data
        X_train1 = np.reshape(X_train1, (-1, inputNum, img_rows, img_cols))
        X_valid1 = np.reshape(X_valid1, (-1, inputNum, img_rows, img_cols))
        X_test1 = np.reshape(X_test1, (-1, inputNum, img_rows, img_cols))

        X_train2 = np.reshape(X_train2, (-1, inputTickdim, inputFeatureDim))
        X_valid2 = np.reshape(X_valid2, (-1, inputTickdim, inputFeatureDim))
        X_test2 = np.reshape(X_test2, (-1, inputTickdim, inputFeatureDim))

        Y_train = to_categorical(y_train, outputClass)
        Y_valid = to_categorical(y_valid, outputClass)
        Y_test = to_categorical(y_test, outputClass)
        print("The output shape is: ")
        print('X_train shape:', X_train1.shape)
        print('X_train shape:', X_train2.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        file.close()
        return X_train1, X_train2, X_valid1, X_valid2, X_test1, X_test2, Y_train, Y_valid, Y_test

import itertools
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class ChallengeEcgMetric:
    ## Compute confusion matrix,How to use
    # cnf_matrix = confusion_matrix(y_test, y_pred)
    # np.set_printoptions(precision=2)
    #
    # # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names,
    #                       title='Confusion matrix, without normalization')
    #
    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix')
    #
    # plt.show()
    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues, saveName="cm.eps"):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='Bilinear', cmap=cmap)#Bilinear,Nearest
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    cm[i][j] = float('%.2f' % cm[i][j])

            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",fontsize=20)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        #plt.show()
        plt.savefig(saveName)

    def WriteScoreRoc(self, scoreDict, rocDict, directoryName=r'./ScoreRoc/', _xlim=[0, 1], _ylim=[0, 1]):
        # rocDict example:
        # rocDict = {}
        # rocList = []
        # rocList.append(predict)
        # rocList.append(target)
        # rocDict["lm"] = rocList
        if not os.path.exists(directoryName):
            os.makedirs(directoryName)
        # write roc curve
        plt.figure(1)
        for itemName in rocDict:
            plt.xlim(_xlim[0], _xlim[1])
            plt.ylim(_ylim[0], _ylim[1])
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(rocDict[itemName][0], rocDict[itemName][1], label=itemName)
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
            plt.savefig(directoryName + '/' + itemName + '.png')
        # write score
        f = open(directoryName + '/' + "output.txt", "w")
        for item in scoreDict:
            f.write(item + "\n")
            f.write(str(scoreDict[item]))
            f.write("\n")
        print("save data and fig")
        f.close()

    def f_score(self, cm):
        # cm=confusion_matrix
        Normal_fscore = 2 * cm[0, 0] / (sum(cm[0]) + sum(cm[:, 0]))
        AF_fscore = 2 * cm[1, 1] / (sum(cm[1]) + sum(cm[:, 1]))
        Other_fscore = 2 * cm[2, 2] / (sum(cm[2]) + sum(cm[:, 2]))
        Noise_fscore = 2 * cm[3, 3] / (sum(cm[3]) + sum(cm[:, 3]))
        all_fscore = (Normal_fscore + AF_fscore + Other_fscore + Noise_fscore) / 4
        three_fscore = (Normal_fscore + AF_fscore + Other_fscore) / 3
        return [all_fscore, three_fscore, Normal_fscore, AF_fscore, Other_fscore,Noise_fscore]
class FIRFilters:
    '''
    This file defines the main physical constants of the system
    '''

    # Speed of sound c=343 m/s
    c = 343.

    # distance to the far field
    ffdist = 10.

    # cut-off frequency of standard high-pass filter
    fc_hp = 300.

    # tolerance for computations
    eps = 1e-10

    def to_16b(self, signal):
        '''
        converts float 32 bit signal (-1 to 1) to a signed 16 bits representation
        No clipping in performed, you are responsible to ensure signal is within
        the correct interval.
        '''
        return ((2 ** 15 - 1) * signal).astype(np.int16)

    def clip(self, signal, high, low):
        '''
        Clip a signal from above at high and from below at low.
        '''
        s = signal.copy()

        s[np.where(s > high)] = high
        s[np.where(s < low)] = low

        return s

    def normalize(selfs, signal):
        sig = signal.copy()
        mean = np.mean(sig)
        std = np.std(sig)
        filtecg = (sig - mean) / std
        return filtecg

    def normalize_DivMax(self, signal, bits=None):
        '''
        normalize to be in a given range. The default is to normalize the maximum
        amplitude to be one. An optional argument allows to normalize the signal
        to be within the range of a given signed integer representation of bits.
        '''

        s = signal.copy()

        s /= np.abs(s).max()

        # if one wants to scale for bits allocated
        if bits is not None:
            s *= 2 ** (bits - 1)
            s = self.clip(signal, 2 ** (bits - 1) - 1, -2 ** (bits - 1))
        return s

    def angle_from_points(self, x1, x2):

        return np.angle((x1[0, 0] - x2[0, 0]) + 1j * (x1[1, 0] - x2[1, 0]))

    def normalize_pwr(self, sig1, sig2):
        '''
        normalize sig1 to have the same power as sig2
        '''

        # average power per sample
        p1 = np.mean(sig1 ** 2)
        p2 = np.mean(sig2 ** 2)

        # normalize
        return sig1.copy() * np.sqrt(p2 / p1)

    def lowpass(self, signal, Fs, fc=fc_hp, plot=False):
        '''
        Filter out the really low frequencies, default is below 50Hz
        '''

        # have some predefined parameters
        rp = 5  # minimum ripple in dB in pass-band
        rs = 60  # minimum attenuation in dB in stop-band
        n = 4  # order of the filter
        type = 'butter'

        # normalized cut-off frequency
        wc = 2. * fc / Fs

        # design the filter
        from scipy.signal import iirfilter, lfilter, freqz
        b, a = iirfilter(n, Wn=wc, rp=rp, rs=rs, btype='lowpass', ftype=type)

        # plot frequency response of filter if requested
        if (plot):
            import matplotlib.pyplot as plt
            w, h = freqz(b, a)

            plt.figure()
            plt.title('Digital filter frequency response')
            plt.plot(w, 20 * np.log10(np.abs(h)))
            plt.title('Digital filter frequency response')
            plt.ylabel('Amplitude Response [dB]')
            plt.xlabel('Frequency (rad/sample)')
            plt.grid()

        # apply the filter
        signal = lfilter(b, a, signal.copy())

        return signal

    def highpass(self, signal, Fs, fc=fc_hp, plot=False):
        '''
        Filter out the really low frequencies, default is below 50Hz
        '''

        # have some predefined parameters
        rp = 5  # minimum ripple in dB in pass-band
        rs = 60  # minimum attenuation in dB in stop-band
        n = 4  # order of the filter
        type = 'butter'

        # normalized cut-off frequency
        wc = 2. * fc / Fs

        # design the filter
        from scipy.signal import iirfilter, lfilter, freqz
        b, a = iirfilter(n, Wn=wc, rp=rp, rs=rs, btype='highpass', ftype=type)

        # plot frequency response of filter if requested
        if (plot):
            import matplotlib.pyplot as plt
            w, h = freqz(b, a)

            plt.figure()
            plt.title('Digital filter frequency response')
            plt.plot(w, 20 * np.log10(np.abs(h)))
            plt.title('Digital filter frequency response')
            plt.ylabel('Amplitude Response [dB]')
            plt.xlabel('Frequency (rad/sample)')
            plt.grid()

        # apply the filter
        signal = lfilter(b, a, signal.copy())

        return signal

    def time_dB(self, signal, Fs, bits=16):
        '''
        Compute the signed dB amplitude of the oscillating signal
        normalized wrt the number of bits used for the signal
        '''

        import matplotlib.pyplot as plt

        # min dB (least significant bit in dB)
        lsb = -20 * np.log10(2.) * (bits - 1)

        # magnitude in dB (clipped)
        pos = self.clip(signal, 2. ** (bits - 1) - 1, 1.) / 2. ** (bits - 1)
        neg = -self.clip(signal, -1., -2. ** (bits - 1)) / 2. ** (bits - 1)

        mag_pos = np.zeros(signal.shape)
        Ip = np.where(pos > 0)
        mag_pos[Ip] = 20 * np.log10(pos[Ip]) + lsb + 1

        mag_neg = np.zeros(signal.shape)
        In = np.where(neg > 0)
        mag_neg[In] = 20 * np.log10(neg[In]) + lsb + 1

        plt.plot(np.arange(len(signal)) / float(Fs), mag_pos - mag_neg)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude [dB]')
        plt.axis('tight')
        plt.ylim(lsb - 1, -lsb + 1)

        # draw ticks corresponding to decibels
        div = 20
        n = int(-lsb / div) + 1
        yticks = np.zeros(2 * n)
        yticks[:n] = lsb - 1 + np.arange(0, n * div, div)
        yticks[n:] = -lsb + 1 - np.arange((n - 1) * div, -1, -div)
        yticklabels = np.zeros(2 * n)
        yticklabels = range(0, -n * div, -div) + range(-(n - 1) * div, 1, div)
        plt.setp(plt.gca(), 'yticks', yticks)
        plt.setp(plt.gca(), 'yticklabels', yticklabels)

        plt.setp(plt.getp(plt.gca(), 'ygridlines'), 'ls', '--')


class cyEcgPlot:
    def readRawArrayData(self, inputNum=1, inputTickdim=1200, inputFeatureDim=20, outputClass=4,
                         fileName='TrainTestRaw.h5'):
        file = h5py.File(fileName, 'r')
        trainData = file['TrainList']
        trainLabel = file['TrainLabelList']

        trainData = trainData.reshape(-1, inputFeatureDim, inputTickdim)
        numTrain = len(trainData)

        def PreProcessing(data, dataLabel):
            print(np.asarray(data).shape)
            numData = len(data)
            data = np.swapaxes(data, 1, 2)
            data = np.asarray(data).reshape((numData, inputTickdim, inputFeatureDim))
            lable = np.asarray(dataLabel, dtype=int).reshape(
                (data.shape[0], 1))  # reshape and translate into interger
            print(data.shape, lable.shape)
            return data, lable

        print("Load from hdf5 file,the shap is ")
        trainData, trainLabel = PreProcessing(trainData, trainLabel)

        # shuffle the data
        trainData, trainLabel = shuffle(trainData, trainLabel, random_state=0)

        X_train = trainData
        y_train = trainLabel

        # reshape data
        X_train = np.reshape(X_train, (-1, inputTickdim, inputFeatureDim))

        Y_train = to_categorical(y_train, outputClass)

        print("The output shape is: ")
        print('X_train shape:', X_train.shape)
        print('y train target', y_train.shape)
        print('Y train target', Y_train.shape)
        file.close()
        return X_train,Y_train
