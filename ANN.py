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

def create_baseline(layers):
	# create model
	model = Sequential()
	model.add(Dense(layers[1], input_dim=layers[0], kernel_initializer='normal', activation='relu'))
	if len(layers) > 2:
		for each in range(len(layers)-2):
			model.add(Dense(layers[each+1], kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def train_and_evaluate_model(model, epoch, X_train, y_train, X_test, y_test):
	# Fitting our model
	model.fit(X_train, y_train, batch_size = 10, nb_epoch = epoch)
	# Predicting the Test set results
	y_pred = model.predict(X_test)
	y_pred = (y_pred > 0.5)
	# Creating the Confusion Matrix
	cm = confusion_matrix(y_test, y_pred)
	print(cm)
	misclassified = np.where(y_test.reshape(len(y_test), 1) != y_pred.reshape(len(y_pred), 1))
	misclassified = X_test[misclassified[0],:]
	return cm, misclassified

def get_misclf_ID(Data, misclassified):
	IDs =[]
	for i in range(len(misclassified)):
		for j in range(len(misclassified[i])):
			ID = (Data.loc[(Data['ClumpTkns'] == misclassified[i][j][0]) & \
				(Data['UnofCSize'] == misclassified[i][j][1]) & \
				(Data['UnofCShape'] == misclassified[i][j][2]) & \
				(Data['MargAdh'] == misclassified[i][j][3]) & \
				(Data['SngEpiCSize'] == misclassified[i][j][4]) & \
				(Data['BareNuc'] == misclassified[i][j][5]) & \
				(Data['BlandCrmtn'] == misclassified[i][j][6]) & \
				(Data['NrmlNuc'] == misclassified[i][j][7]) & \
				(Data['Mitoses'] == misclassified[i][j][8])])
			IDs.append(int(ID.iloc[0]['ID']))
	return IDs



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
#min_max_scaler = preprocessing.MaxAbsScaler()
#x = min_max_scaler.fit_transform(x)


# 4.train model and performed testing using ANN
print("\nSelected Algorithm: ANN")
# evaluate model with standardized dataset

acc_test = []
misclassified = []
n_folds = 5
epoch = 15
skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
for i, (train, test) in enumerate(skf.split(x, y)):
    print ("Running Fold", i+1, "/", n_folds)
    model = None # Clearing the NN.
    model = create_baseline([x.shape[1], 12, 6, 1])
    [cm, misclf] = train_and_evaluate_model(model, epoch, x[train], y[train], x[test], y[test])
    acc = (cm[0,0]+cm[1,1])/np.sum(cm)
    acc_test.append(acc)
    misclassified.append(misclf)

print("\nCross-predicted accuracy: {0:.4f}\n".format(np.mean(acc_test)))

IDs = get_misclf_ID(Data, misclassified)

print(IDs)
#print(len(IDs))
#print(type(IDs[0]))
#print(misclassified)
#print(misclassified[0].shape)

"""
#submission
print("Writing submission.csv file...")
index = [i for i in range(Data.shape[0])]
df2 = pd.DataFrame({'Predictions': predictions}, index=index)
submission = pd.concat([Data, df2], axis=1)
submission.to_csv('wresult.csv', index=False)
"""
