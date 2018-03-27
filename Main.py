#By Hani

#Main

from Models import naive_bayes, logistic_regression, knn, cart, svcm
from LoadDataset import load_dataset

data, target = load_dataset("peersim.csv")



naive_bayes(data, target)
logistic_regression(data, target)
knn(data, target)
cart(data, target)
svcm(data, target)
