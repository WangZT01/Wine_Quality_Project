import unittest
from os import listdir
from os.path import isfile, join

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from DataSets.ImportData import Datasets


class Test(unittest.TestCase):

    def test_createTestData(self):

        address_red = "./DataSets/winequality-red.csv"
        address_white = "./DataSets/winequality-white.csv"

        datasets_red = Datasets(address_red)
        datasets_white = Datasets(address_white)

        X_red, y_red = datasets_red.loadData_origin()
        X_white, y_white = datasets_white.loadData_origin()

        scaler = StandardScaler()
        X_red = scaler.fit_transform(X_red)
        _, X_red_test, _, y_red_test = train_test_split(X_red, y_red, test_size=0.35, random_state=25)

        X_white = scaler.fit_transform(X_white)
        _, X_white_test, _, y_white_test = train_test_split(X_white, y_white, test_size=0.35, random_state=25)

        self.assertEqual(X_red_test.shape[1], 11)
        self.assertEqual(X_white_test.shape[1], 11)
        return X_red_test, y_red_test, X_white_test, y_white_test

    def test_model_number(self):

        model_list_origin = [f for f in listdir('./Model/Train_origindata') if isfile(join('./Model/Train_origindata', f))]

        model_list_binary = [x for x in listdir('./Model/Train_binarydata') if isfile(join('./Model/Train_binarydata', x))]

        self.assertEqual(model_list_origin, model_list_binary)
        return model_list_origin

    def test_model_red(self):
        X_red_test, y_red_test, X_white_test, y_white_test = self.test_createTestData()
        model_list = self.test_model_number()
        address_origin = './Model/Train_origindata/'

        dict_red = dict()
        for model_name in model_list:
            if model_name.find('red') != -1:
                address = address_origin + model_name

                name = model_name.split('_')
                model = joblib.load(address)
                predict = model.predict(X_red_test)
                accuracy = round(accuracy_score(y_red_test, predict) * 100, 2)
                dict_red.update({name[0]: accuracy})
                df_red = pd.DataFrame({'Model': dict_red.keys(), 'Accuracy(%)': dict_red.values()})
        print(df_red)

    def test_model_white(self):
        X_red_test, y_red_test, X_white_test, y_white_test = self.test_createTestData()
        model_list = self.test_model_number()
        address_origin = './Model/Train_origindata/'

        dict_white = dict()
        for model_name in model_list:
            if model_name.find('white') != -1:
                address = address_origin + model_name

                name = model_name.split('_')
                model = joblib.load(address)
                predict = model.predict(X_white_test)
                accuracy = round(accuracy_score(y_white_test, predict) * 100, 2)
                dict_white.update({name[0]: accuracy})
                df_white = pd.DataFrame({'Model': dict_white.keys(), 'Accuracy(%)': dict_white.values()})
        print(df_white)


if __name__ == '__main__':
    unittest.main()