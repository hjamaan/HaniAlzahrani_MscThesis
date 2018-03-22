# By Hani

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score



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

