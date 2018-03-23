#By Hani

from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# load dataset
dataset = datasets.load_iris()

# fitting
model = KNeighborsClassifier()
model.fit(dataset.data, dataset.target)
print(model)

# predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# printing results
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
