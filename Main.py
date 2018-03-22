#By Hani

#Main

from Models import naive_bayes
from LoadDataset import load_dataset

data, target = load_dataset("peersim.csv")



naive_bayes(data, target)
