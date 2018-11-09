# This code is prepared by Onur Barut for Project Assignment 
# of COMP.5450 (FALL 2018) course, UML,  by Dr. Jerome Braun.
#
# References:
# 1- https://www.kaggle.com/charma69/titanic-data-science-solutions/edit

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_val_predict, \
									StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from matplotlib import pyplot as plt
from read_data import *
from sklearn import metrics

def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(15, input_dim=9, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# 1.import data
dataset = 'WBCD.csv'
Data = read_data(dataset)


# 2.Fill with median
Data = fillMed(Data)


#3.data precessing
# select features and Normalization
x=Data.loc[:,['ClumpTkns', 'UnofCSize', 'UnofCShape', 'MargAdh', 'SngEpiCSize',
				'BareNuc', 'BlandCrmtn', 'NrmlNuc', 'Mitoses']].values
y=Data['Malignant'].values

# TRANSFROM GIVES 1 % LESS ACCURATE RESULT!!!!
min_max_scaler = preprocessing.MaxAbsScaler()
x = min_max_scaler.fit_transform(x)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# 4.train model and performed testing using ANN
print("\nSelected Algorithm: ANN")
# evaluate model with standardized dataset
"""
clf = KerasClassifier(build_fn=create_baseline(), epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=np.random.seed(7))
scores = cross_val_score(clf, x, y, cv=kfold)
predictions = cross_val_predict(clf, x, y, cv=kfold)
accuracy = metrics.r2_score(y, predictions)
#print("\nCross-validation scores: {}".format(scores))
print("\nmean training result = {0:.4f}".format(np.mean(scores)))
print("\nCross-predicted accuracy: {0:.4f}\n".format(accuracy))
"""

# Fitting our model 
model = create_baseline()
model.fit(X_train, y_train, batch_size = 10, nb_epoch = 10)
# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
# Creating the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

"""
#submission
print("Writing submission.csv file...")
index = [i for i in range(Data.shape[0])]
df2 = pd.DataFrame({'Predictions': predictions}, index=index)
submission = pd.concat([Data, df2], axis=1)
submission.to_csv('wresult.csv', index=False)
"""