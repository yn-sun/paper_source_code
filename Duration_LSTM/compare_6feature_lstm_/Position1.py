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

directoryName="Predict_s1s2position"
predictAll=np.load(directoryName + '/'+"predict.npy")
targetAll=np.load(directoryName + '/'+"target.npy")
#featureAll=np.load(directoryName + '/'+"feature.npy")
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


###test all
# predict=[]
# target=[]
# p_sum=0.0
# p_s1_sum=0.0
# p_s2_sum=0.0
# for i in range(len(predictAll)):
    # predict=predictAll[i]
    # target=targetAll[i]
    # pre_s2=pretm.get_median_position(predict,2)
    # pre_s1=pretm.get_median_position(predict,0)
    # tar_s2 = pretm.get_median_position(target, 2)
    # tar_s1 = pretm.get_median_position(target, 0)
    # p_s1=pretm.pscore(pre_s1, tar_s1, 1)
    # p_s2=pretm.pscore(pre_s2, tar_s2, 1)
    # print i,p_s1
    # print i,p_s1
    # p_s1_sum=p_s1_sum+p_s1
    # p_s2_sum = p_s2_sum + p_s2
    # p_sum=p_sum+p_s1+p_s2
# p_s2_all = p_s2_sum /len(predictAll)
# p_s1_all = p_s1_sum /len(predictAll)
# p_all=p_sum/len(predictAll)/2
# print 'p_score_all,p_score_s1,p_score_s2,',p_all,p_s1_all,p_s2_all
#print(p_s1+p_s2)


#test class
import argparse
def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='result')
    parser.add_argument('--i', type=int, default=0,help='i*3600')
    return parser.parse_args()

args = get_arguments()

i=args.i
start=i*3600
end=i*3600+5000
predictAll=predictAll[start:end]
targetAll=targetAll[start:end]
p_all,p_s1_all,p_s2_all=pretm.get_all_pscore(predictAll,targetAll,3)
s_all,s_s1_all,s_s2_all=pretm.get_all_sscore(predictAll,targetAll,3)
#print pretm.get_all_pscore(predictAll,targetAll,3)
#print pretm.get_all_sscore(predictAll,targetAll,3)

print ("s_all",round(s_all*100,2))
print ("p_all",round(p_all*100,2))


p=p_s1_all;s=s_s1_all;
f_score=2*p*s/(p+s)
print("f_s1_score",round(f_score*100,2))

p=p_s2_all;s=s_s2_all;
f_score=2*p*s/(p+s)
print("f_s2_score",round(f_score*100,2))
p=p_all;s=s_all;
f_score=2*p*s/(p+s)
print("f_score",round(f_score*100,2))
