# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 19:41:55 2015

@author: insignia
"""

import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics

f = open("/Users/insignia/Desktop/Big_Data_Analytics/Assignment/winequality-white.csv")
f.readline()  # skip the header
data = np.loadtxt(f,delimiter=',')


train, test = cross_validation.train_test_split(data, train_size=0.6, test_size=0.4)
X = train[:, 0:10] 
y = train[:, 11]
A = test[:, 0:10] 
b = test[:, 11]
clf = linear_model.SGDClassifier(n_iter=1000, alpha=0.0001, loss='hinge', penalty='l2')
clf.fit(X,y)
b_pred = clf.predict(A)
train_score= clf.score(X, y)
test_score=clf.score(A,b)
print train_score, test_score
print metrics.confusion_matrix(b, b_pred)
