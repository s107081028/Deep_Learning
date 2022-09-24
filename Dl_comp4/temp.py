# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
import pandas as pd
x_train = np.load('./dataset/x_train_cifar10_unlearn.npy')
y_train = np.load('./dataset/y_train_cifar10.npy')
x_val = np.load('./dataset/x_val_cifar10.npy')
y_val = np.load('./dataset/y_val_cifar10.npy')

y_train = np.argmax(y_train, axis=1)
y_val = np.argmax(y_val, axis=1)
#%%
x_ = []
xv_ = []
for f in x_train:
    x_.append(f.flatten())

for f in x_val:
    xv_.append(f.flatten())
x_ = np.array(x_)
xv_ = np.array(xv_)

#%%
from sklearn import svm
svc=svm.SVC()
from sklearn.metrics import accuracy_score
svc.fit(x_, y_train)
y_pred = svc.predict(xv_)
print(accuracy_score(y_pred, y_val))
#%%
x_test = np.load('./dataset/x_test_cifar10.npy')
test = []
for f in x_test:
    test.append(f.flatten())
test = np.array(test)
pred = svc.predict(test)
#%%
prediction = []
for i in range(len(x_test)):
    prediction.append([])
    prediction[i].append(i)
    prediction[i].append(pred[i])
prediction_out = pd.DataFrame(prediction, columns=['id', 'label'])
prediction_out.to_csv('result_cifar.csv', index=False)