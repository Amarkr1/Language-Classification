# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 19:40:12 2017

@author: akumar47
"""

import pandas as pd
from os import listdir
from os.path import isfile, join
from subprocess import Popen
import subprocess
def compare():
    #file1 = pd.read_csv('predictions_MutlinomialNB_72.csv',dtype={'Category':int})
    path ='C:\\Users\\akumar47\\Dropbox\\Courses\\COMP 551-AML\\Project2\\Language-classification\\ref_files\\'
    path2 = 'C:\\Users\\akumar47\\Dropbox\\Courses\\COMP 551-AML\\Project2\\Language-classification\\predictions\\'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    to_compare = [f for f in listdir(path2) if isfile(join(path2, f))]
    to_compare = [i for i in to_compare if 'LR' in i or 'SVM' in i or 'MultiNB' in i or 'RF' in i or 'LDA' in i or 'En' in i]
#    a = np.array([[0]*len(onlyfiles)]*len(to_compare))
    
    f = open('comparisonFiles.csv','w')
    f.write(' ,')
    for s in onlyfiles:
        f.write('{0} ,'.format(s))
    f.write('\n')
    
    for i,files in enumerate(to_compare):
        file1 = pd.read_csv('{0}{1}'.format(path2,files),dtype={'Category':int})
        f.write("{0},".format(files))
        for j,f2 in enumerate(onlyfiles):
            file2 = pd.read_csv('{0}{1}'.format(path,f2),dtype={'Category':int})
            
            c1 = file1['Category']
            c2 = file2['Category']
            d = c1==c2
            tr = 0
            fal = 0
            for i in d:
                if(i):
                    tr+=1
                else:
                    fal+=1
#            print tr*100.0/(tr+fal)
            f.write("{0},".format(tr*100.0/(tr+fal)))
#            a[i][j] = tr*100.0/(tr+fal)
#            print a
        f.write('\n')
    f.close()
#    return a

if __name__ == '__main__':
    compare()
    p=Popen('comparisonFiles.csv', shell=True)
    subprocess.Popen(r'C:\Program Files (x86)\Microsoft Office\Office16\EXCEL.EXE comparisonFiles.csv')

#    os.system("open -a 'C:/Program Files (x86)/Microsoft Office/Office16/EXCEL.exe' 'comparisonFiles.csv'")
#    print (a)

