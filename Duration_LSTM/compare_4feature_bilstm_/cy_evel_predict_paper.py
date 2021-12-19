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
sns.set_context('paper')
fs=50
displaySecondNum=4

displaySecondNum=4
arrayXlabel=cyMetricLib.getArrayXticks(displaySecondNum,fs)#ax.set_xticks ' arrary,ex [0., fs*1, fs*2, fs*3,fs*4]
arrayXticksLabel=cyMetricLib.getArrayXticksLabel(displaySecondNum,fs)#ax set x label ,ex ax.set_xticklabels(["$0$", r"$\frac{1}{2}\pi$",
                     #r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])

arrayYticksLabel=["S1","S2","systole","diastole"]#ax.set_xticks ' arrary,ex [0., fs*1, fs*2, fs*3,fs*4]
arrayYlabel=np.asarray([0,1,2,3])-5#ax set x label ,ex ax.set_xticklabels(["$0$", r"$\frac{1}{2}\pi$",


directoryName="Predict_s1s2position"
predictAll=np.load(directoryName + '/'+"predict.npy")
targetAll=np.load(directoryName + '/'+"target.npy")
featureAll=np.load(directoryName + '/'+"feature.npy")
sampleNumber=990
predict=predictAll[sampleNumber]
target=targetAll[sampleNumber]
feature=featureAll[sampleNumber]#(3000,200, 6)
feature=feature.squeeze()#(200, 6)
feature=feature.T#(6, 200)
print feature.squeeze().shape
print predict[0:50]
print target[0:50]


print(predict.shape, target.shape)
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
# axes[0,0].plot(predict.squeeze(),'b',linewidth=2.0)
# axes[0,0].plot(target.squeeze(),'r',linewidth=2.0)
# axes[0,0].plot(feature[0],'c',alpha=0.6,label='Homomorphic envelope')
# axes[0,0].plot(feature[1],'m',alpha=0.6,label='hilbert envelope')
# axes[0,0].plot(feature[2],'y',alpha=0.6,label='PSD')
# axes[0,0].plot(feature[3],'g',alpha=0.6,label='wavelet envelope')
# axes[0,0].set_xticks(arrayXlabel)#get the Xlabel position
# axes[0,0].set_xticksLabel(arrayXticksLabel)
# #plt.xticklabels(arrayXticksLabel)#get the x label
# axes[0,0].legend(loc='upper right')
# axes[0,0].set_xlabel('Time(s)')
# axes[0,0].set_ylabel('Amplitude')
# axes[0,0].set_ylim=(1,2)
#
plt.plot(predict.squeeze()-5,'r',alpha=0.8,linestyle='-',linewidth=2.0,label='Heart Cycle')
#plt.plot(target.squeeze(),'r',alpha=0.8,linestyle='--',linewidth=2.0,label='Target Label')
plt.plot(feature[0],alpha=0.6,label='Homomorphic envelope')
plt.plot(feature[1],alpha=0.6,label='hilbert envelope')
plt.plot(feature[2],alpha=0.6,label='PSD')
plt.plot(feature[3],alpha=0.6,label='wavelet envelope')
plt.xticks(arrayXlabel,arrayXticksLabel)#get the Xlabel position
plt.yticks(arrayYlabel,arrayYticksLabel)#get the Xlabel position
plt.legend(loc='upper right')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
# plt.figure()
# plt.plot(feature[4])
plt.show()
pscore =cyMetricLib.pscore(predict,target, 10)
print("pscore",pscore)



