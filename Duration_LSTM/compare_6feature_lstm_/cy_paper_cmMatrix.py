import keras.utils
import numpy as np
import sklearn.metrics as metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cyChallengeLib
cem = cyChallengeLib.ChallengeEcgMetric()
directoryName = "PredictResult"
saveName = "Duration-LSTM"

directoryName="Predict_s1s2position"
predictAll=np.load(directoryName + '/'+"predict.npy")
targetAll=np.load(directoryName + '/'+"target.npy")
featureAll=np.load(directoryName + '/'+"feature.npy")
predict=predictAll.reshape(1,-1)
target=targetAll.reshape(1,-1)

y_test = target.T
y_predict = predict.T
print(y_test.shape, y_predict.shape)
# print(y_test[:20], y_predict[:20])
cnf_matrix = metrics.confusion_matrix(y_test, y_pred=y_predict)
p = metrics.accuracy_score(y_test, y_pred=y_predict)

print(p)
# f1_0 = metrics.f1_score(y_test, y_pred=y_predict, average="micro")
# f1_1 = metrics.f1_score(y_test, y_pred=y_predict, average="macro")
# f1_2 = metrics.f1_score(y_test, y_pred=y_predict, average="macro")
# f1_3 = metrics.f1_score(y_test, y_pred=y_predict, average="macro")
# f1 = (f1_0 + f1_1 + f1_2 + f1_3) / 4
# print("f1 score", f1, f1_0, f1_1, f1_2, f1_3)
# all_fscore=cem.f_score(cnf_matrix)
# print("all_fscore,three_fscore",all_fscore[0],all_fscore[1])
# print(all_fscore)
# plit_confusion_matrix
class_names = ["$S_1$", '$Systole$', "$S_2$", "$Diastole$"]



cem.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False, title='Duration-LSTM Normalized confusion matrix',
                          saveName=saveName + "cm.eps")
plt.figure()
cem.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Duration-LSTM Normalized confusion matrix',
                          saveName=saveName + "cm_nor.eps")
plt.show()
