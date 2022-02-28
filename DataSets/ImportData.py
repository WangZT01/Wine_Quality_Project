import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.neural_network
import torch
import warnings

class Datasets:
    'save dataset\'s location'
    features = []
    def __init__(self, location):
        self.location = location

    def loadData(self):
        ' ".\\DataSets\\winequality-red.csv" '
        with open(self.location, encoding='utf-8') as f:
            self.features = np.loadtxt(f, str, delimiter=";", max_rows=1)
            data = np.loadtxt(f, delimiter=";")
            X_train = data[:, 0:11]
            y_train = data[:, -1]
        return X_train, y_train

    def displayLocation(self):
        print('Location is : %s' % self.location)


    def getFeatures(self):
        print("Features is : %s" % self.features)
        return self.features