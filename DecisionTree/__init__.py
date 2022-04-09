import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from DataSets.ImportData import Datasets
import warnings
warnings.filterwarnings('ignore')

class DecisionTree:
    data = []

    def __init__(self, data):
        self.data = data

    def dt_classifier(self, X_train, y_train, X_test, y_test):
        print(" Decision Tree classifier")

        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        predict = dt.predict(X_test)  # prediction
        print("training rate is :{:.2f}%".format(accuracy_score(y_test, predict) * 100))
        print('score ：{:.2f}'.format(dt.score(X_test, y_test)))
        print(classification_report(predict, y_test))
        print(dt.get_params())
        print(dt)
        return dt

    def normalization(self, X_train):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        return X_train

    def DT_GridSearch(self, X_train, y_train, X_test, y_test):
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9)
        rf = DecisionTreeClassifier()
        parameter_space = {
            'criterion': ["gini", "entropy"],
            'splitter': ['best', 'random'],
            'max_depth': [x for x in range(1, 20, 2)],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        clf = GridSearchCV(rf, parameter_space, n_jobs=-1, cv=5)
        clf.fit(X_train, y_train)  # X is train samples and y is the corresponding labels
        df = pd.DataFrame(clf.cv_results_)

        predict = clf.predict(X_test)
        print("training rate is :{:.2f}%".format(accuracy_score(y_test, predict) * 100))
        print('score ：{:.2f}'.format(clf.score(X_test, y_test)))
        print(classification_report(predict, y_test))

        print('Best score is: ', clf.best_score_)
        print('Best prarmeters is: ', clf.best_params_)

        print(clf.best_estimator_)
        return clf.best_estimator_



def training(address, type):

    location = address
    datasets = Datasets(location)


    if type.startswith('origin'):
        X, y = datasets.loadData_origin()
    elif type.startswith('binary'):
        X, y = datasets.loadData_binary()


    data = datasets.getData()
    dt = DecisionTree(data)
    X = dt.normalization(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 25)
    model = dt.dt_classifier(X_train, y_train, X_test, y_test)
    model_Grid = dt.DT_GridSearch(X_train, y_train, X_test, y_test)

    model_score = model.score(X_test, y_test)
    model_Grid_score = model_Grid.score(X_test, y_test)

    if model_score> model_Grid_score:
        print(model_score)
        #joblib.dump(model, "../Model/Train_origindata/DNN_red_data.pkl")
        plot_confusion_matrix(model, X_test, y_test)
        plt.show()
        return model
    else:
        print(model_Grid_score)
        #joblib.dump(model_Grid, "../Model/Train_origindata/DNN_red_data.pkl")
        plot_confusion_matrix(model_Grid,X_test, y_test)
        plt.show()

        return model_Grid

def save_model(model, data_address, type):
    model_type = 'Train_' + type + 'data'
    if data_address.find('red') != -1:
        model_name = 'DecisionTree_' + 'red' + '_data.pkl'
    if data_address.find('white') != -1:
        model_name = 'DecisionTree_' + 'white' + '_data.pkl'
    model_address = '../Model/' + model_type + '/' + model_name

    joblib.dump(model, model_address)

if __name__ == '__main__':


    address_red = "../DataSets/winequality-red.csv"
    address_white = "../DataSets/winequality-white.csv"

    data_type = ['origin', 'binary']

    for type in data_type:
        model = training(address_red, type)
        save_model(model, address_red, type)
        model = training(address_white, type)
        save_model(model, address_white, type)
