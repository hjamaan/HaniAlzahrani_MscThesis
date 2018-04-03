# By Hani

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression



# Naive Bayes
def naive_bayes(data,target):
    # fit a Naive Bayes model to the data
    model = GaussianNB()
    model.fit(data, target.ravel())
    # Print model info
    print("********************")
    print(model)
    # Predictions
    expected = target
    predicted = model.predict(data)
    # Printing the results from metrics
    print(metrics.classification_report(expected, predicted))
    # Printing the confusion matrix
    print(metrics.confusion_matrix(expected, predicted))
    print("********************")
    kfold = model_selection.KFold(n_splits=10)
    results = model_selection.cross_val_score(model, data, target, cv=kfold)
    print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

    

# Logistic Regression
def logistic_regression(data,target):
    # fit a logistic regression model to the data
    model = LogisticRegression()
    model.fit(data, target.ravel())
    # Print model info
    print("********************")
    print(model)
    # Predictions
    expected = target
    predicted = model.predict(data)
    # Printing the results from metrics
    print(metrics.classification_report(expected, predicted))
    # Printing the confusion matrix
    print(metrics.confusion_matrix(expected, predicted))
    print("********************")


# k-Nearest Neighbor
def knn(data,target):
    # fit a logistic regression model to the data
    model = KNeighborsClassifier()
    model.fit(data, target.ravel())
    # Print model info
    print("********************")
    print(model)
    # Predictions
    expected = target
    predicted = model.predict(data)
    # Printing the results from metrics
    print(metrics.classification_report(expected, predicted))
    # Printing the confusion matrix
    print(metrics.confusion_matrix(expected, predicted))
    print("********************")


# CART
def cart(data,target):
    # fit a logistic regression model to the data
    model = DecisionTreeClassifier()
    model.fit(data, target.ravel())
    # Print model info
    print("********************")
    print(model)
    # Predictions
    expected = target
    predicted = model.predict(data)
    # Printing the results from metrics
    print(metrics.classification_report(expected, predicted))
    # Printing the confusion matrix
    print(metrics.confusion_matrix(expected, predicted))
    print("********************")


# SVM for Classification - SVC
def svcm(data,target):
    # fit a logistic regression model to the data
    model = SVC()
    model.fit(data, target.ravel())
    # Print model info
    print("********************")
    print(model)
    # Predictions
    expected = target
    predicted = model.predict(data)
    # Printing the results from metrics
    print(metrics.classification_report(expected, predicted))
    # Printing the confusion matrix
    print(metrics.confusion_matrix(expected, predicted))
    print("********************")


