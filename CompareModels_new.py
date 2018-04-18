# This module will compare models results
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
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
accuracy = []

aucresults = []
auc = []

f1results = []
f1 = []

precision = []
recall = []

# Specify the N fold
N = 10

for name, model in models:
	names.append(name)
	cv_accuracy = model_selection.cross_val_score(model,  X_input, Y_output, cv=N, scoring='accuracy')
	cv_auc = model_selection.cross_val_score(model,  X_input, Y_output, cv=N, scoring='roc_auc')
	accuracyresults.append(cv_accuracy)
	aucresults.append(cv_auc)
	msg = "%s: %f (%f)" % (name, cv_accuracy.mean(), cv_accuracy.std())
	accuracy.append(cv_accuracy.mean())
	auc.append(cv_auc.mean())
	print('----------------------------------------')
	print(msg)
	Y_pred = cross_val_predict(model,X_input,Y_output,cv=N)
	conf_mat = confusion_matrix(Y_output,Y_pred)
	print(conf_mat)
	print('----------------------------------------')



#Writing to csv
file=open('./result.csv', 'w+')
file.write(' ,')
file.write(str(names))
file.write('\nAccuracy,')
file.write(str(accuracy))
file.write('\nAuc,')
file.write(str(auc))
file.close()



# boxplot for accuracy comparison
graph = plt.figure()
graph.suptitle('Accuracy Comparison')
ax = graph.add_subplot(111)
plt.boxplot(accuracyresults)
ax.set_xticklabels(names)

y_pos = np.arange(len(accuracy))

# bar chart accuracy comparison
graph2 = plt.figure()
graph2.suptitle('Accuracy Comparison')
ax2 = graph2.add_subplot(111)
plt.bar(y_pos, accuracy, align='center', alpha=0.5)
plt.xticks(y_pos, names)
plt.show()



