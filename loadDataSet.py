#By Hani

#importing our own dataset
import numpy as np
import pandas as pd


def loadDataSet(file):
    # comma delimited is the default
    input_file = file
    df = pd.read_csv(input_file, header = 0)
    original_headers = list(df.columns.values)
    df = df._get_numeric_data()
    numeric_headers = list(df.columns.values)
    numpy_array = df.as_matrix()
    fileData = numpy_array[:, [0, 1,2,3,4,5,6]]
    fileTarget = numpy_array[:, [7]]
    print(fileData)
    print(fileTarget)

loadDataSet("peersim.csv")
