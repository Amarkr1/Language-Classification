# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 01:44:25 2017

@author: akumar

Code tested in Python3.6

"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


#class GenerateFeatures(object):
def generateFeatures(filename):
    file_train = pd.read_csv('{0}.csv'.format(filename), sep=',', header= 0 , dtype = {'Id':int ,'Text':str})
    file_train['Text'] = file_train['Text'].str.replace(u'![\u4e00-\u9fff，。／【】、v；‘:\"\",./[]-={}]+' , '')
    file_train['Text'] = file_train['Text'].str.replace('http(.*) ','')
    file_train['Text'] = file_train['Text'].str.replace('URL(.*) ', '')
    file_train['Text'] = file_train['Text'].str.replace(r'(.)\1+', r'\1\1')
    
    #print(file_train['Text'])
    vector = TfidfVectorizer(analyzer='char')
    x = vector.fit_transform(file_train['Text'].values.astype('U')).toarray()
    for i,col in enumerate(vector.get_feature_names()):
        if (ord(col)>=97 and ord(col)<=122) or (ord(col)>=192 and ord(col)<=696):# or (ord(col)>=65 and ord(col)<=90):
            #print(col)
            file_train[col] = x[:, i]
    #        self.alphabet_reference[col] = 0
    features = file_train.drop('Id',axis=1)
    features = features.drop('Text',axis = 1)
    #print(self.alphabet_reference)
    #print(features)
    features.to_csv('{0}_features.csv'.format(filename), sep=',', encoding='utf-8',index=False)
    
def commonFeatures(filename1,filename2):
    df1 = pd.read_csv('{0}.csv'.format(filename1), sep=',',header = 0)
#    print (df1.columns)
    df2 = pd.read_csv('{0}.csv'.format(filename2), sep=',',header = 0)
#    print (df2.columns)
    list1 = list(df1.columns)
    list2 = list(df2.columns)
    common = list(set(list1).intersection(list2))
    remove1 = list(set(list1).difference(common))
    remove2 = list(set(list2).difference(common))
    df1 = df1.drop(remove1,axis=1)
    df2 = df2.drop(remove2,axis=1)
    df1.to_csv('{0}.csv'.format(filename1), sep=',', encoding='utf-8',index = False)
    df2.to_csv('{0}.csv'.format(filename2), sep=',', encoding='utf-8',index = False)

def combine_xy_vectors(filename1,filename2):
    train_x = pd.read_csv('{0}.csv'.format(filename1), sep=',',header = 0)
    train_x = train_x.loc[:, ~train_x.columns.str.contains('^Unnamed')]
    train_y = pd.read_csv('{0}.csv'.format(filename2), sep=',',header = 0)
    train_y = train_y.loc[:, ~train_y.columns.str.contains('^Unnamed')]
    if(len(train_x)==len(train_y)):
        print("Merging x and y vectors according to the ids")
        train_x['category'] = train_y.Category
        train_x.to_csv('trainingSet.csv', sep=',', encoding='utf-8', index = True)
    else:
        print("Vector lengths of x and y are not equal. Not possible to merger")

create = 0

train = 'train_set_x'
test = 'test_set_x'
if(create):
    #generate features for the training set    
    generateFeatures(train)
    #generate features for the training set
    generateFeatures(test)
    #getting common features for making training set and test set in proper format
    #filename1 = train+'_features'
    #filename2 = test+'_features'
    commonFeatures(train+'_features',test+'_features')
    
else:
    print ("Skipping generating features......")

combine_xy_vectors(train+'_features','train_set_y')
