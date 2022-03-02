# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.neural_network
import torch
import warnings
import pandas as pd
from DataSets.ImportData import Datasets


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    df = pd.DataFrame(
        {
            'SepalLength': [6.5, 7.7, 5.1, 5.8, 7.6, 5.0, 5.4, 4.6, 6.7, 4.6],
            'SepalWidth': [3.0, 3.8, 3.8, 2.7, 3.0, 2.3, 3.0, 3.2, 3.3, 3.6],
            'PetalLength': [5.5, 6.7, 1.9, 5.1, 6.6, 3.3, 4.5, 1.4, 5.7, 1.0],
            'PetalWidth': [1.8, 2.2, 0.4, 1.9, 2.1, 1.0, 1.5, 0.2, 2.1, 0.2],
            'Category': [
                'virginica',
                'virginica',
                'setosa',
                'virginica',
                'virginica',
                'versicolor',
                'versicolor',
                'setosa',
                'virginica',
                'setosa'
            ]
        }
    )
    pd.plotting.radviz(df, 'Category')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
