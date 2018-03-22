#By Hani

#importing our own dataset
import numpy as np
import pandas as pd


def load_dataset(csvfile):
    # comma delimited is the default
    input_file = csvfile
    df = pd.read_csv(input_file, header = 0)
    original_headers = list(df.columns.values)
    df = df._get_numeric_data()
    numeric_headers = list(df.columns.values)
    numpy_array = df.as_matrix()
    print
    indexOfArray = numpy_array.shape[1]

    fileData = numpy_array[:, 0:indexOfArray-2]
    fileTarget = numpy_array[:, indexOfArray-1]
    return fileData, fileTarget

