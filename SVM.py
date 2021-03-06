# By Hani
# SVM for classification - SVC

from sklearn import datasets
from sklearn import metrics
from sklearn.svm import SVC

# Load dataset
dataset = datasets.load_iris()

# fit a SVM model to the data

model = SVC()
model.fit(dataset.data, dataset.target)

print(model)

# make predictions

expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
