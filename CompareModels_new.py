# This module will compare models results
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.svm import SVC
from LoadDataset import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
import datetime



# load dataset
dataset = "vssplugin.csv"
X_input, Y_output = load_dataset(dataset)

# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('MLP', MLPClassifier()))
models.append(('SGDC', SGDClassifier()))

# evaluate each model in turn
names = []

accuracyresults = []
accuracy = []

aucresults = []
auc = []

f1results = []
f1 = []

precisionresults = []
precision = []

recallresults = []
recall = []

# Specify the N fold
N = 10

for name, model in models:
	names.append(name)
	cv_accuracy = model_selection.cross_val_score(model,  X_input, Y_output, cv=N, scoring='accuracy')
	cv_auc = model_selection.cross_val_score(model,  X_input, Y_output, cv=N, scoring='roc_auc')
	cv_prec = model_selection.cross_val_score(model,  X_input, Y_output, cv=N, scoring='precision')
	cv_recall = model_selection.cross_val_score(model,  X_input, Y_output, cv=N, scoring='recall')
	cv_f1 = model_selection.cross_val_score(model,  X_input, Y_output, cv=N, scoring='f1')
	accuracyresults.append(cv_accuracy)
	aucresults.append(cv_auc)
	precisionresults.append(cv_prec)
	recallresults.append(cv_recall)
	f1results.append(cv_f1)
	
	msg = "%s: %f (%f)" % (name, cv_accuracy.mean(), cv_accuracy.std())
	accuracy.append(cv_accuracy.mean())
	auc.append(cv_auc.mean())
	precision.append(cv_prec.mean())
	recall.append(cv_recall.mean())
	f1.append(cv_f1.mean())
	
	
	print('----------------------------------------')
	print(msg)
	Y_pred = cross_val_predict(model,X_input,Y_output,cv=N)
	conf_mat = confusion_matrix(Y_output,Y_pred)
	print(conf_mat)
	print('----------------------------------------')


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


#Removing unwantd characters
names = str(names)
names = names.replace('[','').replace(']','').replace("'","")
accuracy = str(accuracy)
accuracy = accuracy.replace('[','').replace(']','').replace("'","")
auc = str(auc)
auc = auc.replace('[','').replace(']','').replace("'","")
precision = str(precision)
precision = precision.replace('[','').replace(']','').replace("'","")
recall = str(recall)
recall = recall.replace('[','').replace(']','').replace("'","")
f1 = str(f1)
f1 = f1.replace('[','').replace(']','').replace("'","")


#Writing to csv
file=open('./result.csv', 'w+')
file.write(dataset +'\n')
file.write(str(N) +'\n')
file.write(' ,')
file.write(str(names))
file.write('\nAccuracy,')
file.write(str(accuracy))
file.write('\nAuc,')
file.write(str(auc))
file.write('\nPrecision,')
file.write(str(precision))
file.write('\nRecall,')
file.write(str(recall))
file.write('\nf1,')
file.write(str(f1))
file.close()



