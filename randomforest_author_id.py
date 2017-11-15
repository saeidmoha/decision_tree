#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
import numpy as np

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print(len(features_train))
for i in range(0,10):
   print(features_train[i])

print(len(labels_train))
for i in range(10):
   print(labels_train[i])


#########################################################
### your code goes here ###
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_jobs=4, n_estimators=40, oob_score=True)
t0 = time()
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")

# Print the feature ranking    
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print ('Feature Ranking: ')
for i in range(10):
    print ("{} feature no.{} ({})".format(i+1,indices[i],importances[indices[i]]))

'''
for index, item in enumerate(importances):
    if item > 0.01:        
        print (index, item)     
'''

pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print("accuracy is =", acc)
print("number of feautures: ", len(features_train[0]))


#########################################################


