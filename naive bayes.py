# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 22:27:24 2017

@author: akumar

Code tested in Python2.7
"""

import csv
import math
import numpy as np
import operator
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cross_validation import train_test_split
#seperate the data by class
def sepByClass(data):
	sep = {}
	for i in range(len(data)):
		vector = data[i]
		if (vector[-1] not in sep):
			sep[vector[-1]] = []
		sep[vector[-1]].append(vector[:-1])
	return sep

dataset = pd.read_csv('train_set_x_features.csv',sep=',',header = 0)
data_values = dataset.values
y_file = pd.read_csv('train_set_y.csv',sep=',', header= 0 , dtype = {'Id':int ,'Category':str})
y = y_file['Category']
train_x, val_x,train_y,val_y = train_test_split(dataset,y,test_size=0.3)

selection = 1
selectBest = SelectKBest(score_func=chi2, k=50)
fit = selectBest.fit(train_x, train_y)
bestFeatures = fit.transform(train_x)
best_indices = selectBest.get_support(indices=True)


if selection==1:
    train_x = bestFeatures
    val_x = np.array(val_x)[:,best_indices]
    train_y = np.array(train_y)

    sep_train = sepByClass([list(x) for x in np.append(train_x,train_y.reshape(train_y.shape[0],1),axis=1)])
    sep_test = sepByClass([list(x) for x in np.append(val_x,val_y.reshape(val_y.shape[0],1),axis=1)])
else:
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    sep_train = sepByClass(map(list,np.append(train_x,train_y.reshape(train_y.shape[0],1),axis=1)))
    sep_test = sepByClass(map(list,np.append(val_x,val_y.reshape(val_y.shape[0],1),axis=1)))


#priors
P_slovak = len(sep_train['0'])/np.float(len(sep_train['0'])+len(sep_train['1'])+len(sep_train['2'])+len(sep_train['3'])+len(sep_train['4']))
P_french = len(sep_train['1'])/np.float(len(sep_train['0'])+len(sep_train['1'])+len(sep_train['2'])+len(sep_train['3'])+len(sep_train['4']))
P_spanish = len(sep_train['2'])/np.float(len(sep_train['0'])+len(sep_train['1'])+len(sep_train['2'])+len(sep_train['3'])+len(sep_train['4']))
P_german = len(sep_train['3'])/np.float(len(sep_train['0'])+len(sep_train['1'])+len(sep_train['2'])+len(sep_train['3'])+len(sep_train['4']))
P_polish = len(sep_train['4'])/np.float(len(sep_train['0'])+len(sep_train['1'])+len(sep_train['2'])+len(sep_train['3'])+len(sep_train['4']))


#calculating mean and standard deviation
def meanStd(array):
    result = [(np.mean(elems),np.std(elems)) for elems in zip(*array)]
#    del result[-1] #to remove the prediction column
    return result

mean_std_class = {} #contains mean and standard deviation of all the features belonging to certain class
for i in range(5):
    mean_std_class['{0}'.format(i)]=meanStd(sep_train['{0}'.format(i)])

prior = [P_slovak, P_french, P_spanish, P_german, P_polish]


def gaussianProbability(x,avg,std):
    if(std==0):
        return 1
    val = math.exp(-(math.pow(x-avg,2)/(2*math.pow(std,2))))
    prob = (1 / (math.sqrt(2*math.pi) * std)) * val
    return prob


def feature_given_class(mean_std,feature):
    prob_class = {}
    for j in range(len(mean_std)):
        prob_class['{0}'.format(j)] = 1
        for k,val in enumerate(mean_std['{0}'.format(j)]):
            mean,std = val
            prob_class['{0}'.format(j)] = prob_class['{0}'.format(j)]*gaussianProbability(feature[k],mean,std)
        prob_class['{0}'.format(j)] = prob_class['{0}'.format(j)]*prior[j]
    return prob_class

def predictClass(features):
    predictions = []
    for feature in features:
#        print len(feature)
        prob_class = feature_given_class(mean_std_class,feature)
#        print prob_class
        predictions.append(max(prob_class.iteritems(), key=operator.itemgetter(1))[0])
    return predictions

def createConfusionMatrix(dictionary):
    cfm = np.zeros((5,5))
    for i in range(len(dictionary)):
        for j in range(len(dictionary['{0}'.format(i)])):
            if(dictionary['{0}'.format(i)][j]=='0'):
                cfm[i][0]+=1
            elif(dictionary['{0}'.format(i)][j]=='1'):
                cfm[i][1]+=1
            elif(dictionary['{0}'.format(i)][j]=='2'):
                cfm[i][2]+=1
            elif(dictionary['{0}'.format(i)][j]=='3'):
                cfm[i][3]+=1
            elif(dictionary['{0}'.format(i)][j]=='4'):
                cfm[i][4]+=1
    return cfm
#inset predictions
nb_val_predict ={}
nb_train_predict ={}

for i in range(5):
#    sep_train_test['{0}'.format(i)] = deleteColumn(sep_train_test['{0}'.format(i)])
    nb_val_predict['{0}'.format(i)] = predictClass(sep_test['{0}'.format(i)])
    nb_train_predict['{0}'.format(i)] = predictClass(sep_train['{0}'.format(i)])

cfm_val = map(list,createConfusionMatrix(nb_val_predict))
cfm_train = map(list,createConfusionMatrix(nb_train_predict))

#prediction on out of bag test set    
test_oob = csv.reader(open('test_set_x_features.csv','r'))
test_oob = list(test_oob)
names_oob = test_oob.pop(0)
if selection==1:
    test_oob = map(list,np.array(test_oob)[:,best_indices])
    
for i in range(len(test_oob)):
    test_oob[i]=[float(x) for x in test_oob[i]]
predictions_oob = predictClass(test_oob)

file_out = open("predictions_NB.csv",'w')
file_out.write('Id,Category\n')
for index,p in enumerate(predictions_oob):
    file_out.write('{0},{1}\n'.format(index,p))   
file_out.close()
    