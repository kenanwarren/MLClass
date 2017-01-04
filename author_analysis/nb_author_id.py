#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.
    Use a Naive Bayes Classifier to identify emails by their authors
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
import sys
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from time import time

features_train, features_test, labels_train, labels_test = preprocess()
clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print("training time: ", round(time() - t0))

t0 = time()
clf.predict(features_test)
print("prediction time: ", round(time() - t0))
print("accuracy: ", clf.score(features_test, labels_test))
