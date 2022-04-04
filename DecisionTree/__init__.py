import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from DataSets.ImportData import Datasets


class DecisionTree:
    data = []

    def __init__(self, data):
        self.data = data

    def splitdataset(self):
        # Separating the target variable
        #'/Users/jinfenghu/COMP 432/COMP6321_Project-master/DataSets/winequality-red.csv'
        with open('/Users/jinfenghu/COMP 432/COMP6321_Project-master/DataSets/winequality-red.csv', encoding='utf-8') as f:
            self.features = np.loadtxt(f, str, delimiter=";", max_rows=1)
            self.data = np.loadtxt(f, delimiter=";")
        X = self.data[:, 0:11]
        Y = self.data[:, -1]
        if "red" in self.location:
            return X, [1 if i >= 6 else 0 for i in Y]
        elif "white" in self.location:
            Y = [0 if i <= 5 else i for i in Y]
            Y = [1 if i == 6 else i for i in Y]
            Y = [2 if i > 6 else i for i in Y]
            return X,Y
        # Splitting the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.35, random_state=25)

        return X, Y, X_train, X_test, y_train, y_test
    # Function to perform training with giniIndex.
    def train_using_gini(X_train, X_test, y_train):
        # Creating the classifier object
        clf_gini = DecisionTreeClassifier(criterion="gini",random_state=100,
                                          max_depth=3, min_samples_leaf=10)

        # Performing training
        clf_gini.fit(X_train, y_train)
        return clf_gini

    # Function to perform training with entropy.
    def tarin_using_entropy(X_train, X_test, y_train):
        # Decision tree with entropy
        clf_entropy = DecisionTreeClassifier(
            criterion="entropy", random_state=100,
            max_depth=3, min_samples_leaf=10)

        # Performing training
        clf_entropy.fit(X_train, y_train)
        return clf_entropy

    # Function to make predictions
    def prediction(X_test, clf_object):
        # Predicton on test with giniIndex
        y_pred = clf_object.predict(X_test)
        print("Predicted values:")
        print(y_pred)
        return y_pred

    # Function to calculate accuracy
    def cal_accuracy(y_test, y_pred):
        print("Confusion Matrix: ",
              confusion_matrix(y_test, y_pred))

        print("Accuracy : ",
              accuracy_score(y_test, y_pred) * 100)

        #print("Report : ",
        #      classification_report(y_test, y_pred))


    # Calling main function
    if __name__ == "__main__":
        # Building Phase
        location_red = '/Users/jinfenghu/COMP 432/COMP6321_Project-master/DataSets/winequality-red.csv'
        datasets_red = Datasets(location_red)
        datasets_red.displayLocation()
        X, y = datasets_red.loadData()
        datasets_red.getFeatures()

        data = datasets_red.getData()
        datasets_red.displayData()
        X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
        clf_gini = train_using_gini(X_train, X_test, y_train)
        clf_entropy = tarin_using_entropy(X_train, X_test, y_train)
        # Operational Phase
        print("Results Using Gini Index:")

        # Operational Phase
        # Prediction using gini
        y_pred_gini = prediction(X_test, clf_gini)
        cal_accuracy(data.y_test, y_pred_gini)

        print("Results Using Entropy:")
        # Prediction using entropy
        y_pred_entropy = prediction(X_test, clf_entropy)
        cal_accuracy(data.y_test, y_pred_entropy)
