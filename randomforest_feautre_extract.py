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
import pickle

np.random.seed(42)
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
#features_train, features_test, labels_train, labels_test = preprocess()
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "rb"))
authors = pickle.load( open(authors_file, "rb") )

from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

# TfidfVectorizer convert a collection of raw documents to a matrix of TF-IDF features.
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train).toarray()
features_test  = vectorizer.transform(features_test).toarray()



print(len(features_train))
print(len(labels_train))
print(len(features_train[0]))

'''
for i in range(0,10):
   print(features_train[i])


for i in range(10):
   print(labels_train[i])
'''

### train on only 150 events just for testing, otherwise it takes for ever.
features_train = features_train[:150]
labels_train   = labels_train[:150]

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
feature_name = vectorizer.get_feature_names()
print ('Feature Ranking: ')
for i in range(10):
    print ("{} feature no.{} ({}) feature name={}".format(i+1,indices[i],importances[indices[i]], feature_name[indices[i]]))

feature_name = vectorizer.get_feature_names()

pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print("accuracy is =", acc)
print("number of feautures: ", len(features_train[0]))


#########################################################


