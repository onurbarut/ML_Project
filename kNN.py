# This code is prepared by Onur Barut for Project Assignment 
# of COMP.5450 (FALL 2018) course, UML,  by Dr. Jerome Braun.
#
# References:
# 1- https://www.kaggle.com/charma69/titanic-data-science-solutions/edit

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score,accuracy_score

from matplotlib import pyplot as plt
from read_data import *
from sklearn import metrics

# 1.import data
dataset = 'WBCD.csv'
Data = read_data(dataset)


# 2.Fill with median
Data = fillMed(Data)


#3.data precessing
# select features and Normalization
x=Data.loc[:,['ClumpTkns', 'UnofCSize', 'UnofCShape', 'MargAdh', 'SngEpiCSize', 'BareNuc', 'BlandCrmtn', 'NrmlNuc', 'Mitoses']]
y=Data['Malignant']
# TRANSFROM GIVES 1 % LESS ACCURATE RESULT!!!!
min_max_scaler = preprocessing.MaxAbsScaler()
x = min_max_scaler.fit_transform(x)


# 4.train model and performed testing using logistic regression
# using SVM
print("\nSelected Algorithm: kNN")
clf=KNeighborsClassifier(n_neighbors = 5)

scores = cross_val_score(clf, x, y, cv=5)
predictions = cross_val_predict(clf, x, y, cv=5)
accuracy = metrics.r2_score(y, predictions)
#print("\nCross-validation scores: {}".format(scores))
print("\nmean training result = {}".format(np.mean(scores)))
print("\nCross-predicted accuracy: {}\n".format(accuracy))

"""
#submission
print("Writing submission.csv file...")
index = [i for i in range(Data.shape[0])]
df2 = pd.DataFrame({'Predictions': predictions}, index=index)
submission = pd.concat([Data, df2], axis=1)
submission.to_csv('wresult.csv', index=False)
"""