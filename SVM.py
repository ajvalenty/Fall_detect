#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 12:11:54 2018

@author: Valenty
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm

df = pd.read_csv('mergedfalls.csv', header=None)
df['fall'] = 1

df2 = pd.read_csv('mergednonfalls.csv', header=None)
df2['fall'] = 0

bigdata = df.append(df2, ignore_index=True)

bigdata = bigdata.fillna(0.0)

bigdata = bigdata.sample(9000)

print(bigdata.shape)
print("If guess is all no")
print(1 - bigdata['fall'].mean())


X = bigdata.iloc[:, :-1].astype(float)
y = bigdata.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

clf = svm.SVC(C=100.0)
clf.fit(X_train_scaled, y_train)  
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#    max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False)

print(clf.score(X_test, y_test))


new_sample = np.random.randn(1, 91)
print(new_sample)

prediction = clf.predict(new_sample)[0]

print("I predict that you just", "fell." if prediction == 1 else "didn't fall.")

