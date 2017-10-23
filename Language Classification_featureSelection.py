# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 22:27:24 2017

@author: akumar

Code tested in Python2.7
"""

#This version is training and val data created after tfidf is done
import pandas as pd
import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from scipy.sparse import *
from scipy import io
import re
import scipy.sparse as sps
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA,TruncatedSVD,SparsePCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression,BayesianRidge
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest,chi2,RFE
import seaborn as sns
import pylab as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#filename = 'train_set_x_small'
#filename = 'train_set_x'
def selectBest(n,X,Y,algorithm):
    if(algorithm==1):
        print ("Algorithm for best feature: K best")
        selectBest = SelectKBest(score_func=chi2, k=n)
        fit = selectBest.fit(X, Y)
        best_indices = selectBest.get_support(indices=True)
        print("Done selecting best features!")
        return fit.transform(X),best_indices
    elif(algorithm==2):
        print ("Algorithm for best feature: RFE")
        model = LogisticRegression()
        rfe = RFE(model,n)
        fit = rfe.fit(X, Y)
        best_indices = rfe.get_support(indices=True)
        print("Done selecting best features!")
        return fit.transform(X),best_indices
    

#if __name__ == '__main__':

m1 = 3000
m2 = 5000
m3 = 32000
#normalizing the parameters
normalize = 0
algo_code = 1
savefiles = 0
writeCode = 1 #whether to write to files or not
select = 1

if(algo_code==1):
    algo = 'Kbest_linear'
elif(algo_code==2):
    algo = "RFE"

        
vector1 = CountVectorizer(analyzer='char',max_features=m1)
vector2 = CountVectorizer(analyzer='char',ngram_range=(2,3),max_features=m2)
#vector3 = CountVectorizer(analyzer='char',ngram_range=(3,3),max_features=m3)
#---------------------------------------
###########Training data################
#------------------------------------
trainX_all = pd.read_csv('train_set_x.csv', sep=',', header= 0 , dtype = {'Id':int ,'Text':str})   
#trainX_all['Text'] = trainX_all['Text'].str.lower()
trainX_all['Text'] = trainX_all['Text'].str.replace(u'![\u4e00-\u9fff，\"\",./[]_-={}]+' , '')
trainX_all['Text'] = trainX_all['Text'].str.replace(u'(\ud83d[\ude00-\ude4f])' , '') #delete emojis
trainX_all['Text'] = trainX_all['Text'].str.replace(u'(\ud83c[\udf00-\uffff])' , '') #symbols and pictographs
trainX_all['Text'] = trainX_all['Text'].str.replace(u'(\ud83d[\u0000-\uddff])' , '') #symbols and pictographs
trainX_all['Text'] = trainX_all['Text'].str.replace(u'(\ud83d[\ude80-\udeff])' , '') #transport and map symbols
trainX_all['Text'] = trainX_all['Text'].str.replace(u'(\ud83c[\udde0-\uddff])' , '') #flags
trainX_all['Text'] = trainX_all['Text'].str.replace(u'[\ufffa-\ufffd]' , '') 
#more_emojis = re.compile(u'('u'\ud83c[\udf00-\udfff]|'u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'u'[\u2600-\u26FF\u2700-\u27BF])+',re.UNICODE)
#trainX_all['Text'] = trainX_all['Text'].str.replace(more_emojis, '')
trainX_all['Text'] = trainX_all['Text'].str.replace(r'[0123456789]', '')
trainX_all['Text'] = trainX_all['Text'].str.replace(r'(\u20AC)', '') #for the euro symbol
#trainX_all['Text'] = trainX_all['Text'].str.replace(r'\xe2\x9d\xa4\xef\xb8\x8f|\xf0\x9f\x98\xa2|\xef\xbf\xbd|\xf0\x9f\x98\xaf|\xf0\x9f\x98\x81', '') #for the euro symbol
#trainX_all['Text'] = trainX_all['Text'].str.replace(r'\xf0\x9f\x99\x8a|\xf0\x9f\x99\x88|\xf0\x9f\x98\x9e|\xf0\x9f\x98\x91|\xf0\x9f\x98\xb3|\xf0\x9f\x98\xad\xf0\x9f\x98\xad', '') 
#trainX_all['Text'] = trainX_all['Text'].str.replace(r'\xf0\x9f\x98\x8a|\xf0\x9f\x98\x85|\xf0\x9f\x98\x90|\xf0\x9f\x91\x8b|\xe2\x99\xa5|\xe2\x96\xa0|\xc3\x83\xc2\xa9', '') 
#trainX_all['Text'] = trainX_all['Text'].str.replace(r'\xc3\x9f|\xf0\x9f\x98\x94|\xe2\x9d\xa4\xef\xb8\x8f|\xf0\x9f\x98\xa2|\xef\xbf\xbd|\xf0\x9f\x98\xaf|\xf0\x9f\x98\x81|\xe2\x9d\xa4', '') #for the euro symbol
#trainX_all['Text'] = trainX_all['Text'].str.replace(r'\xe2\x80\x99|\xc2\xbf|\xf0\x9f\x99\x8a|\xf0\x9f\x99\x88|\xf0\x9f\x98\x9e|\xf0\x9f\x98\x91|\xf0\x9f\x98\xb3|\xf0\x9f\x98\xad\xf0\x9f\x98\xad', '') 
#trainX_all['Text'] = trainX_all['Text'].str.replace(r'\xf0\x9f\x98\x8f|\xf0\x9f\x98\x99|\xc2\xa9|\xc2\xa1|\xf0\x9f\x98\x82|\xe2\x80\xa6|\xf0\x9f\x98\x8a|\xf0\x9f\x98\x85|\xf0\x9f\x98\x90|\xf0\x9f\x91\x8b|\xe2\x99\xa5|\xe2\x96\xa0|\xc3\x83\xc2\xa9', '') 
#trainX_all['Text'] = trainX_all['Text'].str.replace(r'\xf0\x9f\x92\x99|\xf0\x9f\x91\x8c|\xf0\x9f\x92\x9b|\xc2\xab|\xc2\xbb|\xf0\x9f\x94\xa5|\xf0\x9f\x8f\xbb|\xef\xb8\x8f|\xe2\x99\x82|\xf0\x9f\xa4\xb7|\xf0\x9f\x99\x8f|\xf0\x9f\x98\x9c|\xf0\x9f\x92\x95|\xe2\x9c\xa8|\xe2\x80\x93', '') 
#trainX_all['Text'] = trainX_all['Text'].str.replace(r'\xf0\x9f\x91\x8f|\xe2\x82\xac|\xf0\x9f\x94\xb5|\xf0\x9f\x98\x80|\xc2\xaf\xc2\xaf\xe3\x83\x84|\xf0\x9f\xa4\x97', '') 
#trainX_all['Text'] = trainX_all['Text'].str.replace(u'\u2018a|\u2018o|\u2018s|\u2018\u011f|\u201a\xe4|\u201a\u011f|\u201ca|\u201cn|\u201co|\u201ea|\u201eb|\u201ec|\u201ed|\u201ef|\u201eg|\u201eh|\u201ei|\u201ek|\u201el|\u201em|\u201en|\u201eo|\u201ep|\u201es|\u201et|\u201eu|\u201ew|\u2020\u011f|\u2030l|\u2030s|\u2122c|\u2122t|\u221ae|\u221ap|\u221as|\u221a\xae|\u221a\u2122|\U0001f1e7|\U0001f1ea|\U0001f1eb|\U0001f1f7|\U0001f382|\U0001f3b5|\U0001f3b6|\U0001f3ba|\U0001f3fc|\U0001f3fd|\U0001f3fe|\U0001f440|\U0001f449|\U0001f44a|\U0001f44d|\U0001f48b|\U0001f49c|\U0001f4a9|\U0001f4aa|\U0001f54a|\U0001f603|\U0001f604|\U0001f606|\U0001f607|\U0001f609|\U0001f60b|\U0001f60c|\U0001f60d|\U0001f60e|\U0001f612|\U0001f613|\U0001f615|\U0001f618|\U0001f61a|\U0001f61d|\U0001f621|\U0001f629|\U0001f62d|\U0001f62e|\U0001f631|\U0001f634|\U0001f637|\U0001f63b|\U0001f643|\U0001f644|\U0001f64b|\U0001f685|\U0001f914|\U0001f919|\U0001f923|\U0001f924|\U0001f926|\udc4c\udc3d','')
#trainX_all['Text'] = trainX_all['Text'].str.replace(u'\u02c6|\u02dc|\u0301|\u032f|\u035c|\u0361|\u03c3|\u043e|\u0442|\u0ca0|\u200b|\u200d|\u2014|\u2015|\u2018|\u201a|\u201c|\u201d|\u201e|\u2020|\u2021|\u2022|\u2030|\u2039|\u2122|\u221a|\u250a|\u25d5|\u263a|\u2640|\u266b|\u2725|\u30c4|\u4e47|\uf800|\U00011400|\U00011800|\U00011c00|\udc09\udc00|\udc12\udc00|\udc13\udc00|\udc16\udc00|\udc54\udc00|\udc65\udc00|\udc72\udc00|\udc73\udc00|\udcdc\udc00|\udcdd\udc00|\udce2\udc00|\udcec\udc00|\udcef\udc00|\udcf3\udc00|\uddb4\udc00|\uddc0\udc00|\uddcc\udc00|\uddcd\udc00|\uddcf\udc00|\uddd0\udc00|\uddd1\udc00|\uddd2\udc00|\uddd4\udc00|\uddd5\udc00|\uddd6\udc00|\uddd7\udc00|\udddb\udc00|\udddc\udc00|\uddde\udc00|\udde1\udc00|\udde3\udc00|\uddea\udc00|\uddee\udc00|\uddf2\udc00|\uddf3\udc00|\uddf6\udc00|\uddf7\udc00|\uddfa\udc00|\uddfd\udc00|\uddff\udc00|\ude00\udc00|\ude04\udc00|\ude0b\udc00|\ude0c\udc00|\ude0d\udc00|\ude14\udc00|\ude4e\udc00|\udf4b\udc00|\udf7e\udc00|\udf7f\udc00|\udf83\udc00|\udfc5\udc00|\udfc6\udc00|\udfc7\udc00','')
#trainX_all['Text'] = trainX_all['Text'].str.replace(r'[\x92|\x9c|\xa2|\xa4|\xa5|\xa6|\xac|\xad|\xae|\xaf|\xb0|\xb1|\xb2|\xb3|\xb4|\xb6|\xb8|\xba|\xbc]','')
#more_chars = re.compile('\x92|\x9c|[\xa1-\xa9]|[\xaa-\xaf]|[\xb1-\xb9]|[\xba-\xbf]|[\xd1-\xd9]|[\xda-\xdf]|�')
#trainX_all['Text'] = trainX_all['Text'].str.replace(more_chars, '') 
#emojis = re.compile('/u([0-9|#][\x{20E3}])|[\x{00ae}|\x{00a9}|\x{203C}|\x{2047}|\x{2048}|\x{2049}|\x{3030}|\x{303D}|\x{2139}|\x{2122}|\x{3297}|\x{3299}][\x{FE00}-\x{FEFF}]?|[\x{2190}-\x{21FF}][\x{FE00}-\x{FEFF}]?|[\x{2300}-\x{23FF}][\x{FE00}-\x{FEFF}]?|[\x{2460}-\x{24FF}][\x{FE00}-\x{FEFF}]?|[\x{25A0}-\x{25FF}][\x{FE00}-\x{FEFF}]?|[\x{2600}-\x{27BF}][\x{FE00}-\x{FEFF}]?|[\x{2900}-\x{297F}][\x{FE00}-\x{FEFF}]?|[\x{2B00}-\x{2BF0}][\x{FE00}-\x{FEFF}]?|[\x{1F000}-\x{1F6FF}][\x{FE00}-\x{FEFF}]?/u')
urls = re.compile('url[s]?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
https = re.compile('http[s]?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
#rx = re.compile('\w{,15}$')
#trainX_all['Text'] = trainX_all['Text'].str.replace(rx,'')
trainX_all['Text'] = trainX_all['Text'].str.replace(https,'')
#trainX_all['Text'] = trainX_all['Text'].str.replace(https2,'')
trainX_all['Text'] = trainX_all['Text'].str.replace(urls, '')
trainX_all['Text'] = trainX_all['Text'].str.replace(r'(.)\1+', r'\1\1')

#trainX_all['Text'] = trainX_all['Text'].str.replace(' ', '')
trainX_all['Text'] = trainX_all['Text'].fillna('')
trainX_all.to_csv('Train_clean.csv')
trainY_all = pd.read_csv('train_set_y.csv', sep=',', header= 0 , dtype = {'Id':int ,'Category':int}) 
Y = trainY_all['Category']
X_tf = vector1.fit_transform(trainX_all['Text'])
fea_train_tf = ['tf_'+f for f in vector1.get_feature_names()]

X_idf = TfidfTransformer().fit_transform(X_tf)
fea_train_idf = ['idf_'+f for f in vector1.get_feature_names()]

X = sps.hstack((X_tf,X_idf))
fea_train = fea_train_tf+fea_train_idf

X_train,X_val,Y_train,Y_val = train_test_split(X,Y,test_size = 0.3)
X_train_tf,X_val_tf,Y_train_tf,Y_val_tf = train_test_split(X_tf,Y,test_size = 0.3)
X_train_idf,X_val_idf,Y_train_idf,Y_val_idf = train_test_split(X_idf,Y,test_size = 0.3)


#---------------------------------------
###########Testing data OOB################
#------------------------------------
test = pd.read_csv('test_set_x.csv', sep=',', header= 0 , dtype = {'Id':int ,'Text':str})    
test['Text'] = test['Text'].str.lower()
test['Text'] = test['Text'].str.replace(u'![\u4e00-\u9fff，\"\",./[]_-={}]+' , '')
test['Text'] = test['Text'].str.replace(u'(\ud83d[\ude00-\ude4f])' , '') #delete emojis
test['Text'] = test['Text'].str.replace(u'(\ud83c[\udf00-\uffff])' , '') #symbols and pictographs
test['Text'] = test['Text'].str.replace(u'(\ud83d[\u0000-\uddff])' , '') #symbols and pictographs
test['Text'] = test['Text'].str.replace(u'(\ud83d[\ude80-\udeff])' , '') #transport and map symbols
test['Text'] = test['Text'].str.replace(u'(\ud83c[\udde0-\uddff])' , '') #flags
test['Text'] = test['Text'].str.replace(u'[\ufffa-\ufffd]' , '') 
#more_emojis = re.compile(u'('u'\ud83c[\udf00-\udfff]|'u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'u'[\u2600-\u26FF\u2700-\u27BF])+',re.UNICODE)
#test['Text'] = test['Text'].str.replace(more_emojis, '')
test['Text'] = test['Text'].str.replace(r'[0123456789]', '')
test['Text'] = test['Text'].str.replace(r'(\u20AC)', '') #for the euro symbol
#test['Text'] = test['Text'].str.replace(r'\xc3\x9f|\xf0\x9f\x98\x94|\xe2\x9d\xa4\xef\xb8\x8f|\xf0\x9f\x98\xa2|\xef\xbf\xbd|\xf0\x9f\x98\xaf|\xf0\x9f\x98\x81|\xe2\x9d\xa4', '') #for the euro symbol
#test['Text'] = test['Text'].str.replace(r'\xe2\x80\x99|\xc2\xbf|\xf0\x9f\x99\x8a|\xf0\x9f\x99\x88|\xf0\x9f\x98\x9e|\xf0\x9f\x98\x91|\xf0\x9f\x98\xb3|\xf0\x9f\x98\xad\xf0\x9f\x98\xad', '') 
#test['Text'] = test['Text'].str.replace(r'\xf0\x9f\x98\x8f|\xf0\x9f\x98\x99|\xc2\xa9|\xc2\xa1|\xf0\x9f\x98\x82|\xe2\x80\xa6|\xf0\x9f\x98\x8a|\xf0\x9f\x98\x85|\xf0\x9f\x98\x90|\xf0\x9f\x91\x8b|\xe2\x99\xa5|\xe2\x96\xa0|\xc3\x83\xc2\xa9', '') 
#test['Text'] = test['Text'].str.replace(r'\xf0\x9f\x92\x99|\xf0\x9f\x91\x8c|\xf0\x9f\x92\x9b|\xc2\xab|\xc2\xbb|\xf0\x9f\x94\xa5|\xf0\x9f\x8f\xbb|\xef\xb8\x8f|\xe2\x99\x82|\xf0\x9f\xa4\xb7|\xf0\x9f\x99\x8f|\xf0\x9f\x98\x9c|\xf0\x9f\x92\x95|\xe2\x9c\xa8|\xe2\x80\x93', '') 
#test['Text'] = test['Text'].str.replace(r'\xf0\x9f\x91\x8f|\xe2\x82\xac|\xf0\x9f\x94\xb5|\xf0\x9f\x98\x80|\xc2\xaf\xc2\xaf\xe3\x83\x84|\xf0\x9f\xa4\x97', '') 
#emojis = re.compile('/u([0-9|#][\x{20E3}])|[\x{00ae}|\x{00a9}|\x{203C}|\x{2047}|\x{2048}|\x{2049}|\x{3030}|\x{303D}|\x{2139}|\x{2122}|\x{3297}|\x{3299}][\x{FE00}-\x{FEFF}]?|[\x{2190}-\x{21FF}][\x{FE00}-\x{FEFF}]?|[\x{2300}-\x{23FF}][\x{FE00}-\x{FEFF}]?|[\x{2460}-\x{24FF}][\x{FE00}-\x{FEFF}]?|[\x{25A0}-\x{25FF}][\x{FE00}-\x{FEFF}]?|[\x{2600}-\x{27BF}][\x{FE00}-\x{FEFF}]?|[\x{2900}-\x{297F}][\x{FE00}-\x{FEFF}]?|[\x{2B00}-\x{2BF0}][\x{FE00}-\x{FEFF}]?|[\x{1F000}-\x{1F6FF}][\x{FE00}-\x{FEFF}]?/u')
urls = re.compile('url[s]?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
https = re.compile('http[s]?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
#rx = re.compile('\w{,15}$')
#test['Text'] = test['Text'].str.replace(rx,'')
test['Text'] = test['Text'].str.replace(https,'')
#test['Text'] = test['Text'].str.replace(https2,'')
test['Text'] = test['Text'].str.replace(urls, '')
test['Text'] = test['Text'].str.replace(r'(.)\1+', r'\1\1')
test['Text'] = test['Text'].fillna('')#this is a very important feature
test['Text'] = test['Text'].str.replace(' ', '')
X_test_data = test['Text']
test.to_csv('Test_clean.csv')


#vector2 = CountVectorizer(analyzer='char',ngram_range=(2,2),max_features=m2)
X_test_tf = vector1.fit_transform(X_test_data)
fea_test_tf =  ['tf_'+f for f in vector1.get_feature_names()]

X_test_idf = TfidfTransformer().fit_transform(X_test_tf)
fea_test_idf =  ['idf_'+f for f in vector1.get_feature_names()]

X_test = sps.hstack((X_test_tf,X_test_idf))
fea_test = fea_test_tf+fea_test_idf
#X_test_f3 = vector3.fit_transform(X_test_data)
print ("Done building datasets")

#slicing all the dataset to make features ordered
common_all_tf = np.array([i for i in fea_train_tf if (i in fea_test_tf)])
train_idx_tf = [i for i,x in enumerate(fea_train_tf) if x in common_all_tf]
test_idx_tf = [i for i,x in enumerate(fea_test_tf) if x in common_all_tf] 

X_train_tf = X_train_tf.toarray()[:,train_idx_tf]
X_val_tf = X_val_tf.toarray()[:,train_idx_tf]
fea_train_tf = list(np.array(fea_train_tf)[train_idx_tf])
X_test_tf = X_test_tf.toarray()[:,test_idx_tf]
fea_test_tf = list(np.array(fea_test_tf)[test_idx_tf]) #ideally fea_train_tf==fea_test_tf must hold 


common_all_idf = np.array([i for i in fea_train_idf if (i in fea_test_idf)])
train_idx_idf = [i for i,x in enumerate(fea_train_idf) if x in common_all_idf]
test_idx_idf = [i for i,x in enumerate(fea_test_idf) if x in common_all_idf] 

X_train_idf = X_train_idf.toarray()[:,train_idx_idf]
X_val_idf = X_val_idf.toarray()[:,train_idx_idf]
fea_train_idf = list(np.array(fea_train_idf)[train_idx_idf])
X_test_idf = X_test_idf.toarray()[:,test_idx_idf]
fea_test_idf = list(np.array(fea_test_idf)[test_idx_idf]) #ideally fea_train_idf==fea_test_idf must hold 
X_train_idf_all = X_idf[:,train_idx_idf]


common_all = np.array([i for i in fea_train if (i in fea_test)])
train_idx = [i for i,x in enumerate(fea_train_idf) if x in common_all]
test_idx = [i for i,x in enumerate(fea_test_idf) if x in common_all] 

X_train = X_train.toarray()[:,train_idx]
X_val = X_val.toarray()[:,train_idx]
fea_train = list(np.array(fea_train)[train_idx])
X_test = X_test.toarray()[:,test_idx]
fea_test = list(np.array(fea_test)[test_idx]) #ideally fea_train_tf==fea_test_tf must hold 

print ("Done slicing")

if(savefiles==1):
    print ("Saving all the parameters")
    np.savetxt('X_train_idf_307tr.csv',X_train_idf,delimiter=',')
    np.savetxt('X_val_idf_307tr.csv',X_val_idf,delimiter=',')
    np.savetxt('X_test_idf_307tr.csv',X_test_idf,delimiter=',')
    np.savetxt('Y_train_idf_307tr.csv',Y_train_idf,delimiter=',')
    np.savetxt('Y_val_idf_307tr.csv',Y_val_idf,delimiter=',')
    header_training = " ,".join(fea_test_idf).encode('utf-8').strip()
    header_training = header_training+",category"
    y=np.array(Y)
    y = y.reshape(y.shape[0],1)
    x = X_train_idf_all.toarray()
    training = np.append(x,y,axis=1)
    np.savetxt('trainingSet.csv',training,delimiter=',',header = header_training)
else:
    print ("Skipping saving the files")
    
if(select==1):
    n = X_train_tf.shape[1]-50 #selecting 50 features less than the actual one
    X_train_tf_select,best_tf_index = selectBest(n,X_train_tf,Y_train,1)
    X_val_tf_select = X_val_tf[:,best_tf_index]
    X_test_tf_select = X_test_tf [:,best_tf_index]
    
    
    n = X_train_idf.shape[1]-50 #selecting 50 features less than the actual one
    X_train_idf_select,best_idf_index = selectBest(n,X_train_idf,Y_train,1)
    X_val_idf_select = X_val_idf[:,best_idf_index]
    X_test_idf_select = X_test_idf [:,best_idf_index]
    X_train_idf_all_select = X_train_idf_all[:,best_idf_index]
    
    n = X_train.shape[1]-50 #selecting 50 features less than the actual one
    X_train_select,best_index = selectBest(n,X_train,Y_train,1)
    X_val_select = X_val[:,best_index]
    X_test_select = X_test [:,best_index]
    
    X_d1_select = [X_train_tf_select,X_train_idf_select,X_train_select]
    X_d2_select = [X_val_tf_select,X_val_idf_select,X_val_select]
    X_d3_select = [X_test_tf_select,X_test_idf_select,X_test_select]
    #the Y's dont change
    Y_d1_select = [Y_train_tf,Y_train_idf,Y_train]
    Y_d2_select = [Y_val_tf,Y_val_idf,Y_val]
    
    
    print ("Done selecting the best parameters")
else:
    print ("Skipping selection")

if(normalize==1):
    sys.stdout.flush()
    mean_train_tf = np.mean(X_train_tf,axis=0)
    mean_train_idf = np.mean(X_train_idf,axis=0)
    mean_train_tf_idf = np.mean(X_train,axis=0)
    
    X_train_tf,bc_mean_train_tf = np.broadcast_arrays(X_train_tf,mean_train_tf)
    X_train_tf = X_train_tf-bc_mean_train_tf
    X_train_idf,bc_mean_train_idf = np.broadcast_arrays(X_train_idf,mean_train_idf)
    X_train_idf = X_train_idf-bc_mean_train_idf
    X_train,bc_mean_train_tf_idf = np.broadcast_arrays(X_train,mean_train_tf_idf)
    X_train = X_train_tf-bc_mean_train_tf_idf
    
    mean_val_tf = np.mean(X_val_tf,axis=0)
    mean_val_idf = np.mean(X_val_idf,axis=0)
    mean_val_tf_idf = np.mean(X_val,axis=0)
    X_val_tf,bc_mean_val_tf = np.broadcast_arrays(X_val_tf,mean_val_tf)
    X_val_tf = X_val_tf-bc_mean_val_tf
    X_val_idf,bc_mean_val_idf = np.broadcast_arrays(X_val_idf,mean_val_idf)
    X_val_idf = X_val_idf-bc_mean_val_idf
    X_val,bc_mean_val_tf_idf = np.broadcast_arrays(X_val,mean_val_tf_idf)
    X_val = X_val_tf-bc_mean_val_tf_idf
    
    mean_test_tf = np.mean(X_test_tf,axis=0)
    mean_test_idf = np.mean(X_test_idf,axis=0)
    mean_test_tf_idf = np.mean(X_test,axis=0)
    X_test_tf,bc_mean_test_tf = np.broadcast_arrays(X_test_tf,mean_test_tf)
    X_test_tf = X_test_tf-bc_mean_test_tf
    X_test_idf,bc_mean_test_idf = np.broadcast_arrays(X_test_idf,mean_test_idf)
    X_test_idf = X_test_idf-bc_mean_test_idf
    X_test,bc_mean_test_tf_idf = np.broadcast_arrays(X_test,mean_test_tf_idf)
    X_test = X_test_tf-bc_mean_test_tf_idf
    print ("Done subtracting the mean")
else:
    print ("Avoiding normalization parameters")
#common = list(set(header_val).intersection(header_test))



def writeOut(filename,predictions,code):
    path = 'C:\\Users\\akumar47\\Dropbox\\Courses\\COMP 551-AML\\Project2\\Language-classification\\predictions\\'
    if(code==0):
        print("Skipping witing to files")
    else:
        if('select' in filename):
            file_out = open("{0}_{1}({2}).csv".format(path+filename,len(fea_train),algo),'w')
            print ("Written to {0}.csv".format(filename+str(len(fea_train))+algo))
        else:
            file_out = open("{0}_{1}.csv".format(path+filename,len(fea_train),algo),'w')
            print ("Written to {0}.csv".format(filename+str(len(fea_train))))
        file_out.write('Id,Category\n')
        for index,p in enumerate(predictions):
            file_out.write('{0},{1}\n'.format(index,p))   
        file_out.close()
        
        
#running the classifiers

X_d1 = [X_train_tf,X_train_idf,X_train]
X_d2 = [X_val_tf,X_val_idf,X_val]
X_d3 = [X_test_tf,X_test_idf,X_test]
Y_d1 = [Y_train_tf,Y_train_idf,Y_train]
Y_d2 = [Y_val_tf,Y_val_idf,Y_val]
ftype = ['tf','idf','tfidf']
for i in range(3):
    clf = MultinomialNB()
    clf.fit(X_d1[i], Y_d1[i])
    print ('Multinomial Nb:', clf.score(X_d2[i],Y_d2[i]))
    predictions = clf.predict(X_d3[i])
    writeOut("MultiNB_{0}".format(ftype[i]),predictions,writeCode)
    #--------------------------------------------
    clf = LogisticRegression()
    clf.fit(X_d1[i], Y_d1[i])
    #clf.fit(X_train_tfidf, Y_train)
    print ('Logistic Regression: ', clf.score(X_d2[i],Y_d2[i]))
    predictions = clf.predict(X_d3[i])
    writeOut("LR_{0}".format(ftype[i]),predictions,writeCode)
    #--------------------------------------------
    
    clf = svm.LinearSVC()
    clf.fit(X_d1[i], Y_d1[i])
    print ('SVM: ', clf.score(X_d2[i],Y_d2[i]))
    predictions = clf.predict(X_d3[i])
    writeOut("SVM_{0}".format(ftype[i]),predictions,writeCode)
    #--------------------------------------------
    
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_d1[i], Y_d1[i])
    print ('LDA: ', clf.score(X_d2[i],Y_d2[i]))
    predictions = clf.predict(X_d3[i])
    writeOut("LDA_{}".format(ftype[i]),predictions,writeCode)
    
    #--------------------------------------------
    clf = RandomForestClassifier(n_estimators=400)
    clf.fit(X_d1[i], Y_d1[i])
    #clf.fit(X_train_tfidf, Y_train)
    print ('RF: ', clf.score(X_d2[i],Y_d2[i]))
    predictions = clf.predict(X_d3[i])
    writeOut("RF_{0}".format(ftype[i]),predictions,writeCode)
    
    

#--------------------------------------------
    if(select==1):
        clf = MultinomialNB()
        clf.fit(X_d1_select[i], Y_d1[i])
        print ('Multinomial Nb_select:', clf.score(X_d2_select[i],Y_d2[i]))
        predictions = clf.predict(X_d3_select[i])
        writeOut("MultiNB_select_{0}".format(ftype[i]),predictions,writeCode)
        #--------------------------------------------
        clf = LogisticRegression()
        clf.fit(X_d1_select[i], Y_d1[i])
        #clf.fit(X_train_tfidf, Y_train)
        print ('Logistic Regression_select: ', clf.score(X_d2_select[i],Y_d2[i]))
        predictions = clf.predict(X_d3_select[i])
        writeOut("LR_select_{0}".format(ftype[i]),predictions,writeCode)
        #--------------------------------------------
        
        clf = svm.LinearSVC()
        clf.fit(X_d1_select[i], Y_d1[i])
        print ('SVM_select: ', clf.score(X_d2_select[i],Y_d2[i]))
        predictions = clf.predict(X_d3_select[i])
        writeOut("SVM_select_{0}".format(ftype[i]),predictions,writeCode)
        #--------------------------------------------
        
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_d1_select[i], Y_d1[i])
        print ('LDA_select: ', clf.score(X_d2_select[i],Y_d2[i]))
        predictions = clf.predict(X_d3_select[i])
        writeOut("LDA_select_{}".format(ftype[i]),predictions,writeCode)
        
        #--------------------------------------------
        clf = RandomForestClassifier(n_estimators=400)
        clf.fit(X_d1_select[i], Y_d1[i])
        #clf.fit(X_train_tfidf, Y_train)
        print ('RF_select: ', clf.score(X_d2_select[i],Y_d2[i]))
        predictions = clf.predict(X_d3_select[i])
        writeOut("RF_select_{0}".format(ftype[i]),predictions,writeCode)
    

    
    