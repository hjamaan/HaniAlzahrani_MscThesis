# By Hani


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


#peersim = np.loadtxt("peersim.csv", delimiter=",", skiprows=1)



input_file = "peersim.csv"


# comma delimited is the default
df = pd.read_csv(input_file, header = 0)

# for space delimited use:
# df = pd.read_csv(input_file, header = 0, delimiter = " ")

# for tab delimited use:
# df = pd.read_csv(input_file, header = 0, delimiter = "\t")

# put the original column names in a python list
original_headers = list(df.columns.values)

# remove the non-numeric columns
df = df._get_numeric_data()

# put the numeric column names in a python list
numeric_headers = list(df.columns.values)

# create a numpy array with the numeric values for input into scikit-learn
numpy_array = df.as_matrix()

peersimData = numpy_array[:, [0, 1,2,3,4,5,6]]
peersimTarget = numpy_array[:, [7]]


# loading the iris dataset from scikit
dataset = datasets.load_iris()

print("----")
print(peersimData)
print("----")
print(peersimTarget)
# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(peersimData.data, peersimTarget.ravel())

# Predictions
expected = peersimTarget
predicted = model.predict(peersimData)

# Printing the results from metrics
print(metrics.classification_report(expected, predicted))

# Printing the confusion matrix
print(metrics.confusion_matrix(expected, predicted))
