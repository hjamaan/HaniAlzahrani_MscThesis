#By Hani
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


dataset = datasets.load_iris()

#fitting
model = LogisticRegression()
model.fit(dataset.data, dataset.target)

#Printing the model information
print(model)

#predictions
expected = dataset.target
predicted = model.predict(dataset.data)

#Results
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
