import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''

This class is used to import data.
'''

class Datasets:
    'save dataset\'s location'
    features = []
    data = []
    data_panda = pd.DataFrame()
    def __init__(self, location):
        self.location = location

    '''

    This function is used to import the two-class data.
    '''
    def loadData_binary(self):
        ' ".\\DataSets\\winequality-red.csv" '
        with open(self.location, encoding='utf-8') as f:
            self.features = np.loadtxt(f, str, delimiter = ";", max_rows = 1)
            self.data = np.loadtxt(f, delimiter=";")
            X_train = self.data[:, 0:11]
            y_train = self.data[:, -1]

        if "red" in self.location:
            return X_train, [1 if i >= 6 else 0 for i in y_train]
        elif "white" in self.location:
            return X_train, [1 if i >= 6 else 0 for i in y_train]

    '''

    This function is used to import the original data.
    '''
    def loadData_origin(self):
        ' ".\\DataSets\\winequality-red.csv" '
        with open(self.location, encoding='utf-8') as f:
            self.features = np.loadtxt(f, str, delimiter = ";", max_rows = 1)
            self.data = np.loadtxt(f, delimiter=";")
            X_train = self.data[:, 0:11]
            y_train = self.data[:, -1]
        if "red" in self.location:
            return X_train, y_train
        elif "white" in self.location:
            return X_train, y_train

    def loadData_cut_features(self):
        ' ".\\DataSets\\winequality-red.csv" '
        with open(self.location, encoding='utf-8') as f:
            self.features = np.loadtxt(f, str, delimiter = ";", max_rows = 1)
            self.data = np.loadtxt(f, delimiter=";")
            X_train = self.data[:, 0:11]
            y_train = self.data[:, -1]


        X_number = [ np.where(self.features == x) for x in ['alcohol', 'sulphates', 'citric acid', 'volatile acidity']]
        X_train_cut_features = self.data[:, X_number[0][0]]
        X_number = X_number[1:]
        for index in X_number:
            X_train_cut_features = np.append(X_train_cut_features, self.data[:, index[0]], axis=1)

        if "red" in self.location:
            return X_train_cut_features, y_train
        elif "white" in self.location:
            return X_train_cut_features, y_train

    def displayLocation(self):
        print('Location is : %s' % self.location)


    def getFeatures(self):
        print("Features is : %s" % self.features)
        return self.features

    def getData(self):
        return self.data

    '''

    This function is used to transfer the data to dataframe
    '''
    def displayData(self):
        df = pd.DataFrame({
            self.features[0]: self.data[:, 0],
            self.features[1]: self.data[:, 1],
            self.features[2]: self.data[:, 2],
            self.features[3]: self.data[:, 3],
            self.features[4]: self.data[:, 4],
            self.features[5]: self.data[:, 5],
            self.features[6]: self.data[:, 6],
            self.features[7]: self.data[:, 7],
            self.features[8]: self.data[:, 8],
            self.features[9]: self.data[:, 9],
            self.features[10]: self.data[:, 10],
            self.features[11]: self.data[:, 11],

        })

        '''
        pic = pd.plotting.radviz(df,'\"quality\"')
        plt.show()
        '''

        self.data_panda = df
        pd.set_option('display.max_columns', None)
        # 显示所有行
        pd.set_option('display.max_rows', None)
        print(df.describe())
        return df


