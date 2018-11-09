"""
Breast cancer Wisconsin dataset 
@author: sk
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics import roc_auc_score,accuracy_score

'''
1.import data
'''


names=['ID', 'ClumpTkns', 'UnofCSize', 'UnofCShape', 'MargAdh', 
'SngEpiCSize', 'BareNuc', 'BlandCrmtn', 'NrmlNuc', 'Mitoses', 'Class' ]
Data=pd.read_csv('WBCD.csv', names=names)

 


'''
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                 test_size=0.3, random_state=42)
'''

'''
train['BareNuc'].replace(?, np.nan, inplace=true)

'''


Data['BareNuc']=Data['BareNuc'].fillna(Data["BareNuc"].median())

'''
3.data precessing
'''


#select features and Normalization
x=Data.loc[:,['ClumpTkns', 'UnofCSize', 'UnofCShape', 'MargAdh', 'SngEpiCSize', 'BareNuc', 'BlandCrmtn', 'NrmlNuc', 'Mitoses']]
y=Data['Class']


min_max_scaler = preprocessing.MaxAbsScaler()
x_minmax = min_max_scaler.fit_transform(x)



#print(x_minmax)

'''
4.train model and performed testing using crossvalidation
'''


# using SupporVectorMachine


print('\n')
print("SVM")
print('\n')

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, x, y, cv=7)
print(scores)
print("mean = ",np.mean(scores))


'''
print("Using SVC from SVM")

svc = SVC(C = 30, gamma = 0.01)
svc.fit(x,y) 

scores = cross_val_score(svc, x, y ,cv=7)
print(scores)
print("mean = ",np.mean(scores))

'''




#submission

submission = pd.DataFrame({
        "ID": testData["ID"],
        "Class": test_predict
    })
submission.to_csv('wresult.csv', index=False)
