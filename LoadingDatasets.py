#By Hani

'''sklearn has set of datasets, iris and digits are classification datasets'''
from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

'''print the dataset'''
print(iris.data)
print("-----------")


'''in digits it is an array of images, to access each image we can use .images[x]'''
print(digits.images[1])





