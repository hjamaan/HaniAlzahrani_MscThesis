# This module will compare models results
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from LoadDataset import load_dataset

# load dataset
X, Y = load_dataset("peersim.csv")

# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
names = []
accuracyresults = []
f1results = []
precisionresults = []
recallresults = []



for name, model in models:
	kfold = model_selection.KFold(n_splits=10)
	names.append(name)
	cv_accuracy = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
	accuracyresults.append(cv_accuracy)
	msg = "%s: %f (%f)" % (name, cv_accuracy.mean(), cv_accuracy.std())
	print(msg)
	f = open('result.txt','a+')
	f.write(str(name))
	f.write(str(cv_accuracy.mean()))
	f.close()

# boxplot for accuracy comparison
graph = plt.figure()
graph.suptitle('Accuracy Comparison')
ax = graph.add_subplot(111)
plt.boxplot(accuracyresults)
ax.set_xticklabels(names)
plt.show()

