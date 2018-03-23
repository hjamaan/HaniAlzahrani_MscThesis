# Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# load the iris datasets
dataset = datasets.load_iris()
# fitting
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
print(model)
# predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# result
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
