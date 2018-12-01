"""
Breast cancer Wisconsin dataset 
@author: sk
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, \
									StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from matplotlib import pyplot as plt
from read_data import *
from sklearn import metrics

scaler = False

def get_misclf_ID(Data, misclassified, scaler):
	IDs =[]
	for i in range(len(misclassified)):
		for j in range(len(misclassified[i])):
			if not scaler:
				msc = misclassified[i][j]
			else:
				scaler = preprocessing.MinMaxScaler(feature_range = (1,10))
				msc = scaler.fit_transform(misclassified[i][j])
			ID = (Data.loc[
				(Data['ClumpTkns'] == msc[0]) & \
				(Data['UnofCSize'] == msc[1]) & \
				(Data['UnofCShape'] == msc[2]) & \
				(Data['MargAdh'] == msc[3]) & \
				(Data['SngEpiCSize'] == msc[4]) & \
				(Data['BareNuc'] == msc[5]) & \
				(Data['BlandCrmtn'] == msc[6]) & \
				(Data['NrmlNuc'] == msc[7]) & \
				(Data['Mitoses'] == msc[8])
				])
			IDs.append(int(ID.iloc[0]['ID']))
	return IDs

#			ID = (Data.loc[
#				(Data['ClumpTkns'] == msc[0]) & \
#				(Data['UnofCSize'] == msc[1]) & \
#				(Data['UnofCShape'] == msc[2]) & \
#				(Data['MargAdh'] == msc[3]) & \
#				(Data['SngEpiCSize'] == msc[4]) & \
#				(Data['BareNuc'] == msc[5]) & \
#				(Data['BlandCrmtn'] == msc[6]) & \
#				(Data['NrmlNuc'] == msc[7]) & \
#				(Data['Mitoses'] == msc[8])
#				])
#

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
	# Fitting our model
	model.fit(X_train, y_train)
	# Predicting the Test set results
	y_pred = model.predict(X_test)
	y_pred = (y_pred > 0.5)

	# Creating the Confusion Matrix
	cm = confusion_matrix(y_test, y_pred)
	misclassified = np.where(y_test.reshape(len(y_test), 1) != y_pred.reshape(len(y_pred), 1))
	misclassified = X_test[misclassified[0],:]
	return cm, misclassified

# 1.import data
dataset = 'WBCD.csv'
Data = read_data(dataset)


# 2.Fill with median
Data = fillMed(Data)


#3.data precessing
# select features and Normalization
x=Data.loc[:,['ClumpTkns', 'UnofCSize', 'UnofCShape', 'MargAdh', 'SngEpiCSize',
				'BareNuc', 'BlandCrmtn', 'NrmlNuc', 'Mitoses']].values
#x=Data.loc[:,['UnofCSize', 'UnofCShape', 'BareNuc']].values
#x = np.concatenate((x, np.multiply(x[:,5], x[:,6]).reshape(699, 1)), axis=1)
y=Data['Malignant'].values

# TRANSFROM GIVES 1 % LESS ACCURATE RESULT!!!!
# get ID NOT working with scaling!!!!
#min_max_scaler = preprocessing.MaxAbsScaler()
#x = min_max_scaler.fit_transform(x)


# 4.train model and performed testing using Logistic Regression
# using Logistic regression
print("\nSelected Algorithm: Logistic Regression")


acc_test = []
misclassified = []
cms = []
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
for i, (train, test) in enumerate(skf.split(x, y)):
    print ("Running Fold", i+1, "/", n_folds)
    model = None # Clearing the NN.
    model=LogisticRegression(random_state=1,solver='liblinear')
    [cm, misclf] = train_and_evaluate_model(model, x[train], y[train], x[test], y[test])
    acc = (cm[0,0]+cm[1,1])/np.sum(cm)
    cms.append(cm)
    acc_test.append(acc)
    misclassified.append(misclf)

print("\nCross-predicted accuracy: {0:.4f}\n".format(np.mean(acc_test)))
print("\nCross-predicted confusion matrix: \n{}\n".format(sum(cms)))
IDs = get_misclf_ID(Data, misclassified, scaler)
IDs.sort()
print(IDs)

with open('LR_Total misclassified sample IDs.txt', 'w') as file:
	file.write("Cross-predicted accuracy: {0:.4f}\n".format(np.mean(acc_test)))
	file.write("Cross-predicted confusion matrix: \n{}\n".format(sum(cms)))
	for ID in IDs:
		file.write("%d\n" % ID)

print("\nTotal misclassified number of samples: {}\n".format(len(IDs)))
"""
#submission
print("Writing submission.csv file...")
index = [i for i in range(Data.shape[0])]
df2 = pd.DataFrame({'Predictions': predictions}, index=index)
submission = pd.concat([Data, df2], axis=1)
submission.to_csv('wresult.csv', index=False)
"""