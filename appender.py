#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 12:48:13 2018

@author: Valenty
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#
df = pd.read_csv('mergedfalls.csv', header=None)
df['fall'] = 1

df2 = pd.read_csv('mergednonfalls.csv', header=None)
df2['fall'] = 0

bigdata = df.append(df2, ignore_index=True)

bigdata = bigdata.fillna(0.0)

#bigdata = bigdata.sample(7000)

print(bigdata.shape)
print("If guess is all no")
print(1 - bigdata['fall'].mean())


X = bigdata.iloc[:, :-1].astype(float)
y = bigdata.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X.shape[1]))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])



model.fit(X_train, y_train, batch_size=128, epochs=4, verbose=1, validation_split=0.1, shuffle=True)
score = model.evaluate(X_test, y_test, batch_size=128)
print(score)
#model_json = model.to_json()
#with open('model.json', 'w') as json_file:
#    json_file.write(model_json)

#model.save_weights('weights-falls.hdf5')





