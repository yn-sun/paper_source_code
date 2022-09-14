# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 16:08:20 2021

@author: Zhan ao Huang
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

class NN():
    def __init__(self,input_number,hidden_number,output_number,learning_rate):
        self.input_layer=input_number
        self.hidden_layer=hidden_number
        self.output_layer=output_number
        self.learning_rate=learning_rate
        np.random.seed(0)
        self.weight1=np.zeros((self.hidden_layer,self.input_layer)) 
        self.weight2=np.zeros((self.output_layer,self.hidden_layer))
        
        for i in range(len(self.weight1)):
            for j in range(len(self.weight1[i])):
                self.weight1[i][j]=np.random.random()
        for i in range(len(self.weight2)):
            for j in range(len(self.weight2[i])):
                self.weight2[i][j]=np.random.random()
        self.bias1=np.zeros((self.hidden_layer,1))
        self.bias2=np.zeros((self.output_layer,1))

    def sigmoid(self,x):
        return 1.0/(1+math.e**(-x))
        #return 1.0/(1+math.e**(-0.5*x))
    def sigmoid_derivate(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
        #return (0.5*math.e**(-0.5*x))/(1+math.e**(-0.5*x))
    def forward(self,x):
        self.x1=np.dot(self.weight1,x)+self.bias1;
        self.x1_sigmoid=self.sigmoid(self.x1)
        
        self.x2=np.dot(self.weight2,self.x1_sigmoid)+self.bias2
        self.x2_sigmoid=self.sigmoid(self.x2)
        
        return self.x2_sigmoid
    
    def loss(self,x,y):
        loss=0
        for i in range(len(x)):
            self.forward(x[i].reshape(-1,1))
            loss+=((y[i].reshape(-1,1)-self.x2_sigmoid)**2).sum()
        return loss/len(x)
    def F_m(self,x):
        derivate_matrix=np.zeros((len(x),len(x)))
        for i in range(len(x)):
                derivate_matrix[i][i]=self.sigmoid_derivate(x[i][0])
        return derivate_matrix
    def loss_m(self,y):
        return -2*np.dot(self.F_m(self.x2),(y-self.x2_sigmoid))
        #return -2*(y-self.x2_sigmoid)
    def amplitude(self,x0):
        a=0
        for i in range(len(x0)):
            for j in range(len(x0[i])):
                a+=x0[i][j]**2 
        return math.sqrt(a)
    
    def backpropagation(self,x,y):
        
        s2=self.loss_m(y)       
        s1=np.dot(np.dot(self.F_m(self.x1),self.weight2.transpose()),s2)
        
        grad2_weight=np.dot(s2,self.x1_sigmoid.transpose())
        grad1_weight=np.dot(s1,x.transpose())
        grad2_bias=s2
        grad1_bias=s1
        return grad2_weight,grad2_bias,grad1_weight,grad1_bias
    def predict(self,x):
        result=self.forward(x)
        return result
        
    def batch_backpropagation(self,x,y):
        grad2_weight_p0=np.zeros((self.output_layer,self.hidden_layer))
        grad2_bias_p0=np.zeros((self.output_layer,1))
        
        grad2_weight_p1=np.zeros((self.output_layer,self.hidden_layer))
        grad2_bias_p1=np.zeros((self.output_layer,1))
        
        grad1_weight_p0=np.zeros((self.hidden_layer,self.input_layer))
        grad1_bias_p0=np.zeros((self.hidden_layer,1))
        
        grad1_weight_p1=np.zeros((self.hidden_layer,self.input_layer))
        grad1_bias_p1=np.zeros((self.hidden_layer,1))
        
        for i in range(len(x)):    
            self.forward(x[i].reshape(-1,1))    
            gweight2,gbias2,gweight1,gbias1=self.backpropagation(x[i].reshape(-1,1),y[i].reshape(-1,1))
            if y[i][0]==0:
                grad2_weight_p0+=gweight2/len(x)
                grad2_bias_p0+=gbias2/len(x)
                
                grad1_weight_p0+=gweight1/len(x)
                grad1_bias_p0+=gbias1/len(x)
            elif y[i][0]==1:
                grad2_weight_p1+=gweight2/len(x)
                grad2_bias_p1+=gbias2/len(x)
                
                grad1_weight_p1+=gweight1/len(x)
                grad1_bias_p1+=gbias1/len(x)
        
        self.weight2=self.weight2-self.learning_rate*grad2_weight_p0-self.learning_rate*grad2_weight_p1
        self.bias2=self.bias2-self.learning_rate*grad2_bias_p0-self.learning_rate*grad2_bias_p1
        self.weight1=self.weight1-self.learning_rate*grad1_weight_p0-self.learning_rate*grad1_weight_p1
        self.bias1=self.bias1-self.learning_rate*grad1_bias_p0-self.learning_rate*grad1_bias_p1 
    def batch_backpropagation_BWL(self,x,y):
        grad2_weight_p0=np.zeros((self.output_layer,self.hidden_layer))
        grad2_bias_p0=np.zeros((self.output_layer,1))
        
        grad2_weight_p1=np.zeros((self.output_layer,self.hidden_layer))
        grad2_bias_p1=np.zeros((self.output_layer,1))
        
        grad1_weight_p0=np.zeros((self.hidden_layer,self.input_layer))
        grad1_bias_p0=np.zeros((self.hidden_layer,1))
        
        grad1_weight_p1=np.zeros((self.hidden_layer,self.input_layer))
        grad1_bias_p1=np.zeros((self.hidden_layer,1))
        C0,C1=0,0
        for i in range(len(x)):    
            self.forward(x[i].reshape(-1,1))    
            gweight2,gbias2,gweight1,gbias1=self.backpropagation(x[i].reshape(-1,1),y[i].reshape(-1,1))
            if y[i][0]==0:
                grad2_weight_p0+=gweight2
                grad2_bias_p0+=gbias2
                
                grad1_weight_p0+=gweight1
                grad1_bias_p0+=gbias1
                C0+=1
            elif y[i][0]==1:
                grad2_weight_p1+=gweight2
                grad2_bias_p1+=gbias2
                
                grad1_weight_p1+=gweight1
                grad1_bias_p1+=gbias1
                C1+=1
        scale0=C0/(C0+C1)
        scale1=C1/(C0+C1)
        #print(scale0,scale1)
        grad2_weight_p0*=scale0
        grad2_bias_p0*=scale0
        grad1_weight_p0*=scale0
        grad1_bias_p0*=scale0
        
        grad2_weight_p1*=scale1
        grad2_bias_p1*=scale1
        grad1_weight_p1*=scale1
        grad1_bias_p1*=scale1
        
        #print(grad2_weight_p0)
        self.weight2=self.weight2-self.learning_rate*grad2_weight_p0-self.learning_rate*grad2_weight_p1
        self.bias2=self.bias2-self.learning_rate*grad2_bias_p0-self.learning_rate*grad2_bias_p1
        self.weight1=self.weight1-self.learning_rate*grad1_weight_p0-self.learning_rate*grad1_weight_p1
        self.bias1=self.bias1-self.learning_rate*grad1_bias_p0-self.learning_rate*grad1_bias_p1          
    def expand_backpropagation(self,x,y,gamma):
        grad2_weight_p0=np.zeros((self.output_layer,self.hidden_layer))
        grad2_bias_p0=np.zeros((self.output_layer,1))
        
        grad2_weight_p1=np.zeros((self.output_layer,self.hidden_layer))
        grad2_bias_p1=np.zeros((self.output_layer,1))
        
        grad1_weight_p0=np.zeros((self.hidden_layer,self.input_layer))
        grad1_bias_p0=np.zeros((self.hidden_layer,1))
        
        grad1_weight_p1=np.zeros((self.hidden_layer,self.input_layer))
        grad1_bias_p1=np.zeros((self.hidden_layer,1))
        
        for i in range(len(x)):    
            self.forward(x[i].reshape(-1,1))    
            gweight2,gbias2,gweight1,gbias1=self.backpropagation(x[i].reshape(-1,1),y[i].reshape(-1,1))
            if y[i][0]==0:
                grad2_weight_p0+=gweight2/len(x)
                grad2_bias_p0+=gbias2/len(x)
                
                grad1_weight_p0+=gweight1/len(x)
                grad1_bias_p0+=gbias1/len(x)
            elif y[i][0]==1:
                grad2_weight_p1+=gweight2/len(x)
                grad2_bias_p1+=gbias2/len(x)
                
                grad1_weight_p1+=gweight1/len(x)
                grad1_bias_p1+=gbias1/len(x)
        #print('gradient_cosine',(grad2_weight_p0*grad2_weight_p1).sum(),(grad1_weight_p0*grad1_weight_p1).sum())
        #print(self.amplitude(grad2_weight_p1),self.amplitude(grad1_weight_p1))        
        
        count=1e-12
        grad2_weight_p0/=(self.amplitude(grad2_weight_p0)+count)
        grad2_weight_p1/=(self.amplitude(grad2_weight_p1)+count)
        grad2_bias_p0/=(self.amplitude(grad2_bias_p0)+count)
        grad2_bias_p1/=(self.amplitude(grad2_bias_p1)+count)
        
        grad1_weight_p0/=(self.amplitude(grad1_weight_p0)+count)
        grad1_weight_p1/=(self.amplitude(grad1_weight_p1)+count)
        grad1_bias_p0/=(self.amplitude(grad1_bias_p0)+count)
        grad1_bias_p1/=(self.amplitude(grad1_bias_p1)+count)
        
        tmp_grad2_weight_p0=grad2_weight_p0
        tmp_grad2_weight_p1=grad2_weight_p1
        tmp_grad2_bias_p0=grad2_bias_p0
        tmp_grad2_bias_p1=grad2_bias_p1
        
        tmp_grad1_weight_p0=grad1_weight_p0
        tmp_grad1_weight_p1=grad1_weight_p1
        tmp_grad1_bias_p0=grad1_bias_p0
        tmp_grad1_bias_p1=grad1_bias_p1
        
        
        grad2_weight=tmp_grad2_weight_p0+tmp_grad2_weight_p1
        grad2_bias=tmp_grad2_bias_p0+tmp_grad2_bias_p1
        grad1_weight=tmp_grad1_weight_p0+tmp_grad1_weight_p1
        grad1_bias=tmp_grad1_bias_p0+tmp_grad1_bias_p1
        
        grad2_weight/=(self.amplitude(grad2_weight)+count)
        grad2_bias/=(self.amplitude(grad2_bias)+count)
        grad1_weight/=(self.amplitude(grad1_weight)+count)
        grad1_bias/=(self.amplitude(grad1_bias)+count)
        #alpha=0.0
        #print((grad2_weight*tmp_grad2_weight_p0).sum(),(grad1_weight*tmp_grad1_weight_p0).sum())
       
        random_gradient2_weight=np.zeros((self.output_layer,self.hidden_layer))
        for i in range(len(random_gradient2_weight)):
            for j in range(len(random_gradient2_weight[i])):
                random_gradient2_weight[i,j]=np.random.uniform(-1,1)
        random_gradient1_weight=np.zeros((self.hidden_layer,self.input_layer))
        for i in range(len(random_gradient1_weight)):
            for j in range(len(random_gradient1_weight[i])):
                random_gradient1_weight[i,j]=np.random.uniform(-1,1)
        random_gradient2_bias=np.zeros((self.output_layer,1))
        for i in range(len(random_gradient2_bias)):
            for j in range(len(random_gradient2_bias[i])):
                random_gradient2_bias[i,j]=np.random.uniform(-1,1)
        random_gradient1_bias=np.zeros((self.hidden_layer,1))
        for i in range(len(random_gradient1_bias)):
            for j in range(len(random_gradient1_bias[i])):
                random_gradient1_bias[i,j]=np.random.uniform(-1,1)
        
        random_gradient2_weight/=(self.amplitude(random_gradient2_weight)+count)
        random_gradient1_weight/=(self.amplitude(random_gradient1_weight)+count)
        random_gradient2_bias/=(self.amplitude(random_gradient2_bias)+count)
        random_gradient1_bias/=(self.amplitude(random_gradient1_bias)+count)
        
        alpha=np.random.uniform(-0.1,0.1)
        grad2_weight_p0_g=grad2_weight_p0+alpha*random_gradient2_weight
        grad2_bias_p0_g=grad2_bias_p0+alpha*random_gradient2_bias
        grad1_weight_p0_g=grad1_weight_p0+alpha*random_gradient1_weight
        grad1_bias_p0_g=grad1_bias_p0+alpha*random_gradient1_bias
        
        grad2_weight_p0_g/=(self.amplitude(grad2_weight_p0_g)+count)
        grad2_bias_p0_g/=(self.amplitude(grad2_bias_p0_g)+count)
        grad1_weight_p0_g/=(self.amplitude(grad1_weight_p0_g)+count)
        grad1_bias_p0_g/=(self.amplitude(grad1_bias_p0_g)+count)
        
       
        belta=np.random.uniform(-0.1,0.1)
        grad2_weight_p0_g2=-belta*grad2_weight_p0_g+grad2_weight_p0
        grad2_bias_p0_g2=-belta*grad2_bias_p0_g+grad2_bias_p0
        grad1_weight_p0_g2=-belta*grad1_weight_p0_g+grad1_weight_p0
        grad1_bias_p0_g2=-belta*grad1_bias_p0_g+grad1_bias_p0
        
        grad2_weight_p0_g2/=(self.amplitude(grad2_weight_p0_g2)+count)
        grad2_bias_p0_g2/=(self.amplitude(grad2_bias_p0_g2)+count)
        grad1_weight_p0_g2/=(self.amplitude(grad1_weight_p0_g2)+count)
        grad1_bias_p0_g2/=(self.amplitude(grad1_bias_p0_g2)+count)
        
        
        #gamma=np.random.uniform(-1.0,1.0)
        #gamma=-0.9
        grad2_weight_p1=grad2_weight_p1+gamma*(grad2_weight_p1-grad2_weight_p0_g2)
        grad2_bias_p1=grad2_bias_p1+gamma*(grad2_bias_p1-grad2_bias_p0_g2)
        grad1_weight_p1=grad1_weight_p1+gamma*(grad1_weight_p1-grad1_weight_p0_g2)
        grad1_bias_p1=grad1_bias_p1+gamma*(grad1_bias_p1-grad1_bias_p0_g2)
        
        grad2_weight_p1/=(self.amplitude(grad2_weight_p1)+count)
        grad2_bias_p1/=(self.amplitude(grad2_bias_p1)+count)
        grad1_weight_p1/=(self.amplitude(grad1_weight_p1)+count)
        grad1_bias_p1/=(self.amplitude(grad1_bias_p1)+count)
        
        self.weight2=self.weight2-self.learning_rate*((grad2_weight_p0+grad2_weight_p1)/(self.amplitude(grad2_weight_p0+grad2_weight_p1)+count))
        self.bias2=self.bias2-self.learning_rate*((grad2_bias_p0+grad2_bias_p1)/(self.amplitude(grad2_bias_p0+grad2_bias_p1)+count))
        self.weight1=self.weight1-self.learning_rate*((grad1_weight_p0+grad1_weight_p1)/(self.amplitude(grad1_weight_p0+grad1_weight_p1)+count))
        self.bias1=self.bias1-self.learning_rate*((grad1_bias_p0+grad1_bias_p1)/(self.amplitude(grad1_bias_p0+grad1_bias_p1)+count)) 
        
    
    def evaluate_auc(self,x,y):
        y_convert=[]
        for i in range(len(y)):
            if y[i][0]==1:#positive [1,0]
                y_convert.append(1)
            else:
                y_convert.append(0)
        p_score=[]
        for i in range(len(x)):
            p=self.softmax(self.predict(x[i].reshape(-1,1)))
            if y_convert[i]==0:#[0,1]
                p_score.append(p[0])
            else:#[1,0]
                p_score.append(p[0])
        #print(p_score)      
        auc=roc_auc_score(y_convert,p_score)
        return auc
    def evaluate_f1_score(self,x,y):
        y_predict=[]
        y_true=[]
        for i in range(len(y)):
            if y[i][0]==1:#positive [1,0]
                y_true.append(1)
            else:
                y_true.append(0)
        for i in range(len(x)):
            p=self.softmax(self.predict(x[i].reshape(-1,1)))
            y_predict.append(np.argmax(p))
        f1=f1_score(y_true,y_predict)
        return f1
    def evaluate_confusion_matrix(self,x,y):
        y_predict=[]
        y_true=[]
        for i in range(len(y)):
            if y[i][0]==1:#positive [1,0]
                y_true.append(1)
            else:
                y_true.append(0)
        for i in range(len(x)):
            p=self.softmax(self.predict(x[i].reshape(-1,1)))
            y_predict.append(np.argmax(p))
        cmatrix=confusion_matrix(y_true,y_predict)
        #print(cmatrix)
        TN,FP,FN,TP=cmatrix.ravel()
        
        sen,spe,ppr,acc=0,0,0,0
        if (TP+FN)!=0:
            sen=TP/(TP+FN)
        if (TN+FP)!=0:
            spe=TN/(TN+FP)
        if (TP+FP)!=0:
            ppr=TP/(TP+FP)
        if (TP+TN+FP+FN)!=0:
            acc=(TP+TN)/(TP+TN+FP+FN)
        TPR,FPR=0,0
        if (TP+FN)!=0:
            TPR=TP/(TP+FN)
        if (FP+TN)!=0:
            FPR=TN/(FP+TN)
        
        macc=(TPR+FPR)/2
        g_mean=(TPR*FPR)**0.5
        return acc,ppr,sen,spe,macc,g_mean
    def softmax(self,x):
        #a/b
        b=0
        for i in range(len(x)):
            b+=math.e**(x[i])
        for i in range(len(x)):
            x[i]=math.e**x[i]/b
        return x
    def ROC(self,x,y):
        predict=[]
        TPR=[]
        FPR=[]
        for i in range(len(x)):
            #0 for p<0.5; 1 for p>0.5
            p=self.softmax(self.predict(x[i].reshape(-1,1)))
            predict.append(p[1])#[1,0] for positive,[0,1] for negative
        for i in np.arange(0,1.1,0.1):
            TP,FP,TN,FN=0,0,0,0
            
            for j in range(len(predict)):
                if predict[j]>i:#predict x[i] as negative [0,1]
                    if y[j][0]<y[j][1]:
                        TN+=1
                    else:
                        FN+=1
                else:
                    if y[j][0]<y[j][1]:
                        FP+=1
                    else:
                        TP+=1
            #sen=TP*1.0/(TP+FN)
            FPR.append(FP*1.0/(FP+TN))
            TPR.append(TP*1.0/(TP+FN))
            
        plt.plot(FPR,TPR)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC')
        plt.show()
        return np.array(FPR),np.array(TPR)
