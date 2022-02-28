import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.neural_network
import torch
import warnings

from DataSets.ImportData import Datasets


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    location = ".\\DataSets\\winequality-red.csv"
    datasets = Datasets(location)
    datasets.displayLocation()
    X_train, y_train = datasets.loadData()
    datasets.getFeatures()
    print(X_train)
    print(y_train)