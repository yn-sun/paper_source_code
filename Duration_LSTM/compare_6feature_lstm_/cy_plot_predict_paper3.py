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
#sns.set_style("whitegrid")
fs=50
displaySecondNum=4

arrayXlabel=cyMetricLib.getArrayXticks(displaySecondNum,fs)#ax.set_xticks ' arrary,ex [0., fs*1, fs*2, fs*3,fs*4]
arrayXticksLabel=cyMetricLib.getArrayXticksLabel(displaySecondNum,fs)#ax set x label ,ex ax.set_xticklabels(["$0$", r"$\frac{1}{2}\pi$",
                     #r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])
arrayYticksLabel=["S1","Systole","S2","Diastole"]#ax.set_xticks ' arrary,ex [0., fs*1, fs*2, fs*3,fs*
arrayYlabel=np.asarray([0,1,2,3])#ax set x label ,ex ax.set_xticklabels(["$0$", r"$\frac{1}{2}\pi$",

directoryName="Predict_s1s2position"
predictAll=np.load(directoryName + '/'+"predict.npy")
targetAll=np.load(directoryName + '/'+"target.npy")
featureAll=np.load(directoryName + '/'+"feature.npy")

# print(predict.shape, target.shape)
plt.figure(figsize=(20,8))
plt.subplot(411)
sampleNumber=2633
predict=predictAll[sampleNumber]
target=targetAll[sampleNumber]
feature=featureAll[sampleNumber]#(3000,200, 6)
feature=feature.squeeze()#(200, 6)
feature=feature.T#(6, 200)

plt.plot(predict.squeeze(),'b',alpha=0.8,linestyle='-',linewidth=2.0,label='Predict tagging')
plt.plot(target.squeeze(),'r',alpha=0.8,linestyle='--',linewidth=2.0,label='Target tagging')
plt.plot(feature[0],'c',alpha=0.8,label='Homomorphic envelope',linewidth=0.7)
plt.plot(feature[1],'m',alpha=0.8,label='Hilbert envelope',linewidth=0.7)
plt.plot(feature[2],'y',alpha=0.8,label='PSD',linewidth=0.7)
plt.plot(feature[3],'g',alpha=0.8,label='Wavelet envelope',linewidth=0.7)
plt.xticks(arrayXlabel,arrayXticksLabel)#get the Xlabel position
plt.yticks(arrayYlabel,arrayYticksLabel)#get the Xlabel position
plt.legend(loc='upper right')

plt.ylabel('Amplitude')
arrayYlabel=np.asarray([0,1,2,3])*2#ax set x label ,ex ax.set_xticklabels(["$0$", r"$\frac{1}{2}\pi$",
plt.subplot(412)
sampleNumber=204
predict=predictAll[sampleNumber]
target=targetAll[sampleNumber]
feature=featureAll[sampleNumber]#(3000,200, 6)
feature=feature.squeeze()#(200, 6)
feature=feature.T#(6, 200)
plt.plot(predict.squeeze()*2,'b',alpha=0.8,linestyle='-',linewidth=2.0,label='Predict Label')
plt.plot(target.squeeze()*2,'r',alpha=0.8,linestyle='--',linewidth=2.0,label='Target Label')
plt.plot(feature[0],'c',alpha=0.8,label='Homomorphic envelope',linewidth=0.7)
plt.plot(feature[1],'m',alpha=0.8,label='hilbert envelope',linewidth=0.7)
plt.plot(feature[2],'y',alpha=0.8,label='PSD',linewidth=0.7)
plt.plot(feature[3],'g',alpha=0.8,label='wavelet envelope',linewidth=0.7)
plt.xticks(arrayXlabel,arrayXticksLabel)#get the Xlabel position
plt.yticks(arrayYlabel,arrayYticksLabel)#get the Xlabel position
#plt.legend(loc='upper right')
plt.ylabel('Amplitude')

plt.subplot(413)
sampleNumber=500
predict=predictAll[sampleNumber]
target=targetAll[sampleNumber]
feature=featureAll[sampleNumber]#(3000,200, 6)
feature=feature.squeeze()#(200, 6)
feature=feature.T#(6, 200)
plt.plot(predict.squeeze(),'b',alpha=0.8,linestyle='-',linewidth=2.0,label='Predict Label')
plt.plot(target.squeeze(),'r',alpha=0.8,linestyle='--',linewidth=2.0,label='Target Label')
plt.plot(feature[0],'c',alpha=0.8,label='Homomorphic envelope',linewidth=0.7)
plt.plot(feature[1],'m',alpha=0.8,label='hilbert envelope',linewidth=0.7)
plt.plot(feature[2],'y',alpha=0.8,label='PSD',linewidth=0.7)
plt.plot(feature[3],'g',alpha=0.8,label='wavelet envelope',linewidth=0.7)
plt.xticks(arrayXlabel,arrayXticksLabel)#get the Xlabel position
plt.yticks(arrayYlabel,arrayYticksLabel)#get the Xlabel position
#plt.legend(loc='upper right')

plt.ylabel('Amplitude')

plt.subplot(414)
sampleNumber=1500
predict=predictAll[sampleNumber]
target=targetAll[sampleNumber]
feature=featureAll[sampleNumber]#(3000,200, 6)
feature=feature.squeeze()#(200, 6)
feature=feature.T#(6, 200)
plt.plot(predict.squeeze(),'b',alpha=0.8,linestyle='-',linewidth=2.0,label='Predict Label')
plt.plot(target.squeeze(),'r',alpha=0.8,linestyle='--',linewidth=2.0,label='Target Label')
plt.plot(feature[0],'c',alpha=0.8,label='Homomorphic envelope',linewidth=0.7)
plt.plot(feature[1],'m',alpha=0.8,label='hilbert envelope',linewidth=0.7)
plt.plot(feature[2],'y',alpha=0.8,label='PSD',linewidth=0.7)
plt.plot(feature[3],'g',alpha=0.8,label='wavelet envelope',linewidth=0.7)
plt.xticks(arrayXlabel,arrayXticksLabel)#get the Xlabel position
plt.yticks(arrayYlabel,arrayYticksLabel)#get the Xlabel position
#plt.legend(loc='upper right')

plt.ylabel('Amplitude')
#plt.show()
plt.savefig('6feature_output.pdf',dpi = 1000,bbox_inches='tight')


