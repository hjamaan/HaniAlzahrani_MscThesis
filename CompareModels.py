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
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# load dataset
X, Y = load_dataset("peersim.csv")


validation_size = 0.40
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)



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
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	names.append(name)
	cv_accuracy = model_selection.cross_val_score(model,  X_train, Y_train, cv=kfold, scoring='accuracy')
	accuracyresults.append(cv_accuracy)
	msg = "%s: %f (%f)" % (name, cv_accuracy.mean(), cv_accuracy.std())
	print('----------------------------------------')
	print(msg)
	f = open('result.txt','a+')
	f.write(str(name))
	f.write(str(cv_accuracy.mean()))
	f.close()
	print('----------------------------------------')

for name, model in models:
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        print(model)
        print(accuracy_score(Y_validation, predictions))
        print(confusion_matrix(Y_validation, predictions))
        print(classification_report(Y_validation, predictions))
        print('----------------------------------------')


# boxplot for accuracy comparison
graph = plt.figure()
graph.suptitle('Accuracy Comparison')
ax = graph.add_subplot(111)
plt.boxplot(accuracyresults)
ax.set_xticklabels(names)
plt.show()

