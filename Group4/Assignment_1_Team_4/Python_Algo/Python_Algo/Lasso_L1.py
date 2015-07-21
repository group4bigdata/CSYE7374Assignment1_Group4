# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 05:35:33 2015

@author: insignia
"""

import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import preprocessing

f = open("/Users/insignia/Desktop/Big_Data_Analytics/Assignment/winequality-white_multi.csv")
f.readline()  # skip the header
data = np.loadtxt(f,delimiter=';')


train, test = cross_validation.train_test_split(data, train_size=0.6, test_size=0.4)
X = train[:, 0:10] 
scaler = preprocessing.StandardScaler().fit(X)

X = scaler.transform(X)
y = train[:, 11]
A = test[:, 0:10] 
A = scaler.transform(A)
b = test[:, 11]
clf = linear_model.Lasso(alpha=.0001, random_state=np.random, normalize=True)
clf.fit(X,y)
# The mean square error
print ("Residual sum of squares: %.2f" %np.mean((clf.predict(A) - b) ** 2))
# Explained variance score: 1 is perfect prediction
print ('Variance score: %.2f' % clf.score(A, b))
