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
sampleNumber=499
predict=predictAll[sampleNumber]
target=targetAll[sampleNumber]
#test one
pretm=cyMetricLib.HeartSoundEvaluation()
# p_s2=pretm.get_median_position(predict,2)
# p_s1=pretm.get_median_position(predict,0)
# print(p_s1+p_s2)
# t_s2=pretm.get_median_position(target,2)
# t_s1=pretm.get_median_position(target,0)
# print(t_s1+t_s2)
# print pretm.pscore(p_s2,t_s2,1)


# ###test all
# predict=[]
# target=[]
# p_sum=0.0
# p_s1_sum=0.0
# p_s2_sum=0.0
# for i in range(len(predictAll)):
#     predict=predictAll[i]
#     target=targetAll[i]
#     pre_s2=pretm.get_median_position(predict,2)
#     pre_s1=pretm.get_median_position(predict,0)
#     tar_s2 = pretm.get_median_position(target, 2)
#     tar_s1 = pretm.get_median_position(target, 0)
#     p_s1=pretm.pscore(pre_s1, tar_s1, 1)
#     p_s2=pretm.pscore(pre_s2, tar_s2, 1)
#     print i,p_s1
#     print i,p_s1
#     p_s1_sum=p_s1_sum+p_s1
#     p_s2_sum = p_s2_sum + p_s2
#     p_sum=p_sum+p_s1+p_s2
# p_s2_all = p_s2_sum /len(predictAll)
# p_s1_all = p_s1_sum /len(predictAll)
# p_all=p_sum/len(predictAll)/2
# print 'p_score_all,p_score_s1,p_score_s2,',p_all,p_s1_all,p_s2_all
# #print(p_s1+p_s2)


#test class
p_sum=0.0
p_all,p_s1_all,p_s2_all=pretm.get_all_pscore(predictAll,targetAll,5)
s_all,s_s1_all,s_s2_all=pretm.get_all_sscore(predictAll,targetAll,5)
print ("s_all",s_all)
print ("p_all",p_all)


p=p_s1_all;s=s_s1_all;
f_score=2*p*s/(p+s)
print("f_s1_score",f_score)

p=p_s2_all;s=s_s2_all;
f_score=2*p*s/(p+s)
print("f_s2_score",f_score)
p=p_all;s=s_all;
f_score=2*p*s/(p+s)
print("f_score",f_score)