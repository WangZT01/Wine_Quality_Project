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
import seaborn as sns

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    location_red = ".\\DataSets\\winequality-red.csv"
    red_data = pd.read_csv(location_red)
    sns.countplot(data = red_data, x = 'quality')
    plt.show()

    plt.figure()
    location_white = ".\\DataSets\\winequality-white.csv"
    white_data = pd.read_csv(location_white)
    sns.countplot(data = white_data, x = 'quality')
    plt.show()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
