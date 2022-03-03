import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Datasets:
    'save dataset\'s location'
    features = []
    data = []
    def __init__(self, location):
        self.location = location

    def loadData(self):
        ' ".\\DataSets\\winequality-red.csv" '
        with open(self.location, encoding='utf-8') as f:
            self.features = np.loadtxt(f, str, delimiter = ";", max_rows = 1)
            self.data = np.loadtxt(f, delimiter=";")
            X_train = self.data[:, 0:11]
            y_train = self.data[:, -1]
        return X_train, y_train

    def displayLocation(self):
        print('Location is : %s' % self.location)


    def getFeatures(self):
        print("Features is : %s" % self.features)
        return self.features

    def getData(self):
        return self.data

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
        plt.figure(figsize = (20,8))
        pd.plotting.parallel_coordinates(df, 'quality', color = ['b', 'g', 'r', 'c', 'm', 'y', 'k'])
        plt.show()

        plt.figure(figsize = (20,8))
        sns.countplot(data=df, x='quality')
        plt.show()


