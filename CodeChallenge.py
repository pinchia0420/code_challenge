# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 11:46:40 2017

@author: 陳
"""

import numpy as np
import os, os.path, re
import scipy.io.wavfile as wavfile
import librosa
from sklearn import svm
from sklearn import decomposition
from sklearn import neighbors
## loading  training dataset
path = 'C://Users//陳/Desktop//Code Challenge//umich_ds_cc_2017-master//train_data'
data = []
names = []
labels = []
types = 10
for filename in os.listdir(path):
    rate, dataval = wavfile.read(path+"//"+filename)  ## rates are all the same
    data.append(dataval)
    names.append(filename)
for name in names:
    ind = name.find('_')
    name = name[:ind]
    labels.append(name)
data = np.asarray(data)

##  loading testing dataset
path_test = 'C://Users//陳/Desktop//Code Challenge//umich_ds_cc_2017-master//test_data'
data_test = []
names_test = []
for testname in os.listdir(path_test):
    rate, dataval = wavfile.read(path_test+"//"+testname)  ## rates are all the same
    data_test.append(dataval)
    names_test.append(testname)
data_test = np.asarray(data_test)

##  extract training mfcc
features = []
for i in range(len(data)):
    mfcc = librosa.feature.mfcc(y=data[i], sr=rate)
    aver = np.mean(mfcc, axis = 1)
    features.append(aver.reshape(20))
    
##  extract testing mfcc
features_test = []
for i in range(len(data_test)):
    mfcc = librosa.feature.mfcc(y=data_test[i], sr=rate)
    aver = np.mean(mfcc, axis = 1)
    features_test.append(aver.reshape(20))

##  train data
svc = svm.LinearSVC()
svc.fit(features, labels)
svc_score = svc.score(features, labels)
predict_svc = svc.predict(features_test)
knn = neighbors.KNeighborsClassifier(n_neighbors=types, metric='minkowski')
knn.fit(features, labels)
knn_score = knn.score(features,labels) ## knn seems to work better
predict_knn = knn.predict(features_test)

##  Save .csv file
#Since knn seems to perform better in training, this project will submit prediction
# of knn result
with open('pinchia.csv', 'w') as f:
    for ind in range(len(predict_knn)):
        filename = names_test[ind]
        predict = predict_knn[ind]
        line = filename + "," + predict + "\n"
        f.write(line)