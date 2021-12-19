#metric our experiment and record it in file
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from madmom.evaluation.beats import calc_absolute_errors,BeatIntervalError

def WriteScoreRoc(scoreDict,rocDict,directoryName =r'./ScoreRoc/',_xlim=[0,1],_ylim=[0,1]):
    #rocDict example:
    # rocDict = {}
    # rocList = []
    # rocList.append(predict)
    # rocList.append(target)
    # rocDict["lm"] = rocList
    if not os.path.exists(directoryName):
        os.makedirs(directoryName)
    #write roc curve
    plt.figure(1)
    for itemName in rocDict:
        plt.xlim(_xlim[0],_xlim[1])
        plt.ylim(_ylim[0],_ylim[1])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(rocDict[itemName][0], rocDict[itemName][1], label=itemName)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig(directoryName+'/'+itemName+'.png')
    #write score
    f = open(directoryName+'/'+"output.txt", "w")
    for item in scoreDict:
        f.write(item+ "\n")
        f.write(str(scoreDict[item]))
        f.write("\n")
    print("save data and fig")
    f.close()

class HeartSoundEvaluation:
# class HeartSoundEvaluation(BeatEvaluation):
#     def p_s
# evaluation functions for beat detection
    def pscore(self,detections, annotations, toleranceWindows=10):
        """
        Calculate the P-score accuracy for the given detections and annotations.

        The P-score is determined by taking the sum of the cross-correlation
        between two impulse trains, representing the detections and annotations
        allowing for a tolerance of 20% of the median annotated interval [1]_.

        Parameters
        ----------
        detections : list or numpy array
            Detected beats.
        annotations : list or numpy array
            Annotated beats.
        tolerance : float, optional
            Evaluation tolerance (fraction of the median beat interval).

        Returns
        -------
        pscore : float
            P-Score.

        Notes
        -----
        Contrary to the original implementation which samples the two impulse
        trains with 100Hz, we do not quantise the annotations and detections but
        rather count all detections falling withing the defined tolerance window.

        References
        ----------
        .. [1] M. McKinney, D. Moelants, M. Davies and A. Klapuri,
               "Evaluation of audio beat tracking and music tempo extraction
               algorithms",
               Journal of New Music Research, vol. 36, no. 1, 2007.

        """
        # neither detections nor annotations are given, perfect score
        if len(detections) == 0 and len(annotations) == 0:
            return 1.
        # either beat detections or annotations are empty, score 0
        if (len(detections) == 0) != (len(annotations) == 0):
            return 0.
        # at least 2 annotations must be given to calculate an interval
        # if len(annotations) < 2:
        #     raise BeatIntervalError("At least 2 annotations are needed for"
        #                             "P-Score.")

        # tolerance must be greater than 0
        if float(toleranceWindows) <= 0:
            raise ValueError("`tolerance` must be greater than 0.")

        # make sure the annotations and detections have a float dtype
        detections = np.asarray(detections, dtype=np.float)
        annotations = np.asarray(annotations, dtype=np.float)

        # the error window is the given fraction of the median beat interval
        window = toleranceWindows
        # errors
        errors = calc_absolute_errors(detections, annotations)
        #print errors
        # count the instances where the error is smaller or equal than the window
        p = len(detections[errors <= window])
        # normalize by the max number of detections/annotations
        p /= float(max(len(detections), len(annotations)))
        # return p-score
        return p


    def sensitive_score(self, detections, annotations, toleranceWindows=10):
        """
        Calculate the P-score accuracy for the given detections and annotations.

        The P-score is determined by taking the sum of the cross-correlation
        between two impulse trains, representing the detections and annotations
        allowing for a tolerance of 20% of the median annotated interval [1]_.

        Parameters
        ----------
        detections : list or numpy array
            Detected beats.
        annotations : list or numpy array
            Annotated beats.
        tolerance : float, optional
            Evaluation tolerance (fraction of the median beat interval).

        Returns
        -------
        pscore : float
            P-Score.

        Notes
        -----
        Contrary to the original implementation which samples the two impulse
        trains with 100Hz, we do not quantise the annotations and detections but
        rather count all detections falling withing the defined tolerance window.

        References
        ----------
        .. [1] M. McKinney, D. Moelants, M. Davies and A. Klapuri,
               "Evaluation of audio beat tracking and music tempo extraction
               algorithms",
               Journal of New Music Research, vol. 36, no. 1, 2007.

        """
        # neither detections nor annotations are given, perfect score
        if len(detections) == 0 and len(annotations) == 0:
            return 1.
        # either beat detections or annotations are empty, score 0
        if (len(detections) == 0) != (len(annotations) == 0):
            return 0.
        # at least 2 annotations must be given to calculate an interval
        # if len(annotations) < 2:
        #     raise BeatIntervalError("At least 2 annotations are needed for"
        #                             "P-Score.")

        # tolerance must be greater than 0
        if float(toleranceWindows) <= 0:
            raise ValueError("`tolerance` must be greater than 0.")

        # make sure the annotations and detections have a float dtype
        detections = np.asarray(detections, dtype=np.float)
        annotations = np.asarray(annotations, dtype=np.float)

        # the error window is the given fraction of the median beat interval
        window = toleranceWindows
        # errors
        errors = calc_absolute_errors(detections, annotations)
        # print errors
        # count the instances where the error is smaller or equal than the window
        p = len(detections[errors <= window])
        # normalize by the max number of detections/annotations
        #p /= float(min(len(detections), len(annotations)))
        p /= float(len(annotations))
        # return p-score
        return p

    def _find_s2Position(self, y_predict, label=2):
        # predict list,get the start_position,and end position
        # input y_predict = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 1, 1, 2, 2, 2, 2, 3, 3, 1, 1, 2, 2, 2, 2,
        #             3, 3]
        # label=2
        # return:[[6, 7], [14, 17], [22, 25]]
        positionList = []
        index = np.where(y_predict == label)#return turble
        index = index[0]
        if len(index)==0:#index ==NULL
            index=np.zeros((1,1))
        min_bandary = index[0]
        max_bandary = index[0]
        for i in range(len(index) - 1):  # iterator all index array
            if (index[i + 1] - index[i] == 1):  # the index is continious
                min_bandary = min_bandary
                max_bandary = index[i + 1]
            else:  # the index is continious,index[i+1]-index[i]!=1
                positionList.append([min_bandary, max_bandary])
                min_bandary = index[i + 1]
                max_bandary = index[i + 1]
        return positionList

    #using method  of average windows to process After convolutional
    def _find_s2Position_convolution(self, y_predict, label=2):
        # predict list,get the start_position,and end position
        # input y_predict = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 1, 1, 2, 2, 2, 2, 3, 3, 1, 1, 2, 2, 2, 2,
        #             3, 3]
        # label=2
        # return:[[6, 7], [14, 17], [22, 25]]
        positionList = []
        ##convolutional#@###################
        #w = np.array([0.25, 0.25, 0.25, 0.25])
        #w = np.array([0.2, 0.2, 0.2, 0.2,0.2])
        w = np.array([0.33, 0.33, 0.33])
        y_predict_conv_float=np.convolve(y_predict,w,'same')#After convolutional
        y_predict_conv = np.around(y_predict_conv_float)
        y_predict_conv=y_predict_conv.astype(int)

        ##
        index = np.where(y_predict_conv == label)  # return turble
        index = index[0]
        if len(index) == 0:  # index ==NULL
            index = np.zeros((1, 1))
        min_bandary = index[0]
        max_bandary = index[0]
        for i in range(len(index) - 1):  # iterator all index array
            if (index[i + 1] - index[i] == 1):  # the index is continious
                min_bandary = min_bandary
                max_bandary = index[i + 1]
            else:  # the index is continious,index[i+1]-index[i]!=1
                positionList.append([min_bandary, max_bandary])
                min_bandary = index[i + 1]
                max_bandary = index[i + 1]
        return positionList

    def _get_median_position_predict(self, positionList):
        # input:[[8, 11], [33, 37], [58, 61], [82, 85], [106, 109], [130, 134], [154, 158]]
        # output:[9, 35, 59, 83, 107, 132, 156]
        median_positionList = []
        for i in positionList:
            # if((i[1] - i[0])<1):
                #print((i[1] - i[0]))
                # continue
            median = i[0] + (i[1] - i[0]) / 2
            median_positionList.append(median)
        return median_positionList
    def _get_median_position_target(self, positionList):
        # input:[[8, 11], [33, 37], [58, 61], [82, 85], [106, 109], [130, 134], [154, 158]]
        # output:[9, 35, 59, 83, 107, 132, 156]
        median_positionList = []
        for i in positionList:
            # if((i[1] - i[0])<1):
            #     continue
            median = i[0] + (i[1] - i[0]) / 2
            median_positionList.append(median)
        return median_positionList


    def get_median_position_predict(self, y_predict, label,convolution=True):
        if(convolution!=True):
            positionList = self._find_s2Position(y_predict, label=label)
        else:
            positionList = self._find_s2Position_convolution(y_predict, label=label)#convolution
        return self._get_median_position_predict(positionList)
    def get_median_position_target(self, y_predict, label,convolution=False):
        if (convolution != True):
            positionList = self._find_s2Position(y_predict, label=label)
        else:
            positionList = self._find_s2Position_convolution(y_predict, label=label)#convolution
        return self._get_median_position_target(positionList)
    def get_all_pscore(self,predictAll,targetAll,toleranceWindows=3,convolution=False):
        #read test_s1s2Position.py
        predict = []
        target = []
        p_sum = 0.0
        p_s1_sum = 0.0
        p_s2_sum = 0.0
        for i in range(len(predictAll)):
            predict = predictAll[i]
            target = targetAll[i]
            if(len(predict)<200):
                predict=predict[25:-25]
                target = target[25:-25]
            elif(200<=len(predict)<300):
                predict=predict[50:-50]
                target = target[50:-50]
            else:
                predict=predict[100:-100]
                target = target[100:-100]
            pre_s2 = self.get_median_position_predict(predict, 2,convolution)
            pre_s1 = self.get_median_position_predict(predict, 0,convolution)

            tar_s2 = self.get_median_position_target(target, 2,convolution=False)
            tar_s1 = self.get_median_position_target(target, 0,convolution=False)
            p_s1 = self.pscore(pre_s1, tar_s1, toleranceWindows)
            p_s2 = self.pscore(pre_s2, tar_s2, toleranceWindows)
            #print i, p_s1
            #print i, p_s1
            p_s1_sum = p_s1_sum + p_s1
            p_s2_sum = p_s2_sum + p_s2
            p_sum = p_sum + p_s1 + p_s2
            p_score_s2 = p_s2_sum / len(predictAll)
            p_score_s1 = p_s1_sum / len(predictAll)
        p_all = p_sum / len(predictAll) / 2
        return p_all,p_score_s1,p_score_s2


    def get_all_sscore(self, predictAll, targetAll, toleranceWindows=3,convolution=True):
        # read test_s1s2Position.py
        predict = []
        target = []
        p_sum = 0.0
        p_s1_sum = 0.0
        p_s2_sum = 0.0
        for i in range(len(predictAll)):
            predict = predictAll[i]
            target = targetAll[i]
            pre_s2 = self.get_median_position_predict(predict, 2,convolution)
            pre_s1 = self.get_median_position_predict(predict, 0,convolution)

            tar_s2 = self.get_median_position_target(target, 2,convolution=False)
            tar_s1 = self.get_median_position_target(target, 0,convolution=False)
            p_s1 = self.sensitive_score(pre_s1, tar_s1, toleranceWindows)
            p_s2 = self.sensitive_score(pre_s2, tar_s2, toleranceWindows)
            #print i, p_s1
            #print i, p_s1
            p_s1_sum = p_s1_sum + p_s1
            p_s2_sum = p_s2_sum + p_s2
            p_sum = p_sum + p_s1 + p_s2
        p_score_s2 = p_s2_sum / len(predictAll)
        p_score_s1 = p_s1_sum / len(predictAll)
        p_all = p_sum / len(predictAll) / 2
        return p_all, p_score_s1, p_score_s2




#generate the xlabel position
def getArrayXticks(displaySecondNum=3,fs=4000):
    arrayXticks=[]
    for i in range(0,displaySecondNum+1):
        arrayXticks.append(i*fs);
    return arrayXticks
#get the Xlabel
def getArrayXticksLabel(displaySecondNum=3,fs=4000):
    arrayXticksLabel=[]
    for i in range(0,displaySecondNum+1):
        strs=str(i)+' s'
        arrayXticksLabel.append(strs);
    return arrayXticksLabel


