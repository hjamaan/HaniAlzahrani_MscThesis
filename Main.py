#By Hani

#Main

from Models import naive_bayes, logistic_regression, knn
from LoadDataset import load_dataset

data, target = load_dataset("peersim.csv")



naive_bayes(data, target)
logistic_regression(data, target)
knn(data, target)
