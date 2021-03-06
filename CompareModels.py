# This module will compare models results
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.svm import SVC
from LoadDataset import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


# load dataset
X_input, Y_output = load_dataset("peersim.csv")

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
	names.append(name)
	cv_accuracy = model_selection.cross_val_score(model,  X_input, Y_output, cv=10, scoring='accuracy')
	accuracyresults.append(cv_accuracy)
	msg = "%s: %f (%f)" % (name, cv_accuracy.mean(), cv_accuracy.std())
	print('----------------------------------------')
	print('', msg)
	y_pred = cross_val_predict(model,X_input,Y_output,cv=10)
	print('y pred',y_pred)
	conf_mat = confusion_matrix(Y_output,y_pred)
	print(conf_mat)
	f = open('result.txt','a+')
	f.write(str(name))
	f.write(str(cv_accuracy.mean()))
	f.close()
	print('----------------------------------------')

'''for name, model in models:
        model.fit(X_input, Y_output)
        predictions = model.predict(X_validation)
        print(model)
        print(accuracy_score(Y_validation, predictions))
        print(confusion_matrix(Y_validation, predictions))
        print(classification_report(Y_validation, predictions))
        print('----------------------------------------')'''
kf = cross_validation.KFold(len(Y_output), n_folds=5)
for train_index, test_index in kf:
        X_train, X_test = X_input[train_index], X_input[test_index]
        y_train, y_test = Y_output[train_index], Y_output[test_index]
        model.fit(X_train, y_train)
        matrix = confusion_matrix(y_test, model.predict(X_test))
        print(metrics.classification_report(y_test, model.predict(X_test)))
        print(matrix)


# boxplot for accuracy comparison
graph = plt.figure()
graph.suptitle('Accuracy Comparison')
ax = graph.add_subplot(111)
plt.boxplot(accuracyresults)
ax.set_xticklabels(names)
plt.show()

