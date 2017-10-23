# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 00:03:13 2017

@author: akumar47
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 18:28:44 2017

@author: akumar47
"""
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.svm import SVR
from sklearn import linear_model
from collections import Counter

path = "C:\\Users\\akumar47\\Dropbox\\Courses\\COMP 551-AML\\Project2\\Language-classification\\ensamble prediction\\"
onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) if '.csv' in f]
predict_file = open(path+"out Ensamble Prediction_freqHigh.csv",'w')
predict_file.write('Id,Category\n')
predict_file2 = open(path+"out Ensamble Prediction_weighted.csv",'w')
predict_file2.write('Id,Category\n')
array = np.array([])
w = []
for file in onlyfiles:
    if('out' in file):
        continue
    else:
        
        p1 = pd.read_csv(path+file, sep=',', header= 0 , dtype = {'Id':int ,'Category':int}) 
        l1 = np.array(p1['Category'])
        if(len(array)==0):
            array = l1
        else:
            array = np.vstack([array,l1])
        w = w + [float(file.replace('.csv','').split('_')[-1])/100000]
            
for i in range(array.shape[1]):
    dec = list(array[:,i])
    scr = [0.0,0.0,0.0,0.0,0.0]
    for j in range(len(w)):
        if(dec[j]==0):
            scr[0]+=w[j]
        elif(dec[j]==1):
            scr[1]+=w[j]
        elif(dec[j]==2):
            scr[2]+=w[j]
        elif(dec[j]==3):
            scr[3]+=w[j]
        elif(dec[j]==4):
            scr[4]+=w[j]
        
    most_common,num_most_common = Counter(dec).most_common(1)[0]
    predict_file.write('{0},{1}\n'.format(i,most_common))
    predict_file2.write('{0},{1}\n'.format(i,scr.index(max(scr))))
    
predict_file.close()
predict_file2.close()
print('Ensamble predictions done!')