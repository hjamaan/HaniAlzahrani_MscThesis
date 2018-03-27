from sklearn.model_selection import KFold
from Models import naive_bayes, logistic_regression, knn, cart
from LoadDataset import load_dataset

data, target = load_dataset("peersim.csv")


kf = KFold(n_splits=10)
kf.get_n_splits(data)

print(kf)  

for train_index, test_index in kf.split(data):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = data[train_index], data[test_index]
   y_train, y_test = target[train_index], target[test_index]
