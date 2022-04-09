import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from DataSets.ImportData import Datasets
from RandomForest import RandomForest
import pandas as pd
from os import listdir
from os.path import isfile, join

class Test_Model:


    model_red = set()
    model_white = set()
    def __init__(self, model_list, data_type):
        self.model_list = model_list
        self.data_type = str(data_type)

        for model in self.model_list:

            if model.find('red') != -1:
                self.model_red.add(model)
            else:
                self.model_white.add(model)

    def createTestData(self):

        address_red = "../DataSets/winequality-red.csv"
        address_white = "../DataSets/winequality-white.csv"

        datasets_red = Datasets(address_red)
        datasets_white = Datasets(address_white)
        if self.data_type.startswith('origin'):
            X_red, y_red = datasets_red.loadData_origin()
            X_white, y_white = datasets_white.loadData_origin()

        elif self.data_type.startswith('binary'):
            X_red, y_red = datasets_red.loadData_binary()
            X_white, y_white = datasets_white.loadData_binary()

        scaler = StandardScaler()
        X_red = scaler.fit_transform(X_red)
        _, X_red_test, _, y_red_test = train_test_split(X_red, y_red, test_size=0.35, random_state=25)

        X_white = scaler.fit_transform(X_white)
        _, X_white_test, _, y_white_test = train_test_split(X_white, y_white, test_size=0.35, random_state=25)

        return X_red_test, y_red_test, X_white_test, y_white_test

    def printScore(self):

        X_red_test, y_red_test, X_white_test, y_white_test = self.createTestData()
        print('-----------------------------------------------------------')
        print("model's score of the " + self.data_type +  " wine data")
        print('-----------------------------------------------------------')
        print("Model for red Mine")
        dict_red = dict()
        for model_name in list(self.model_red):

            model_address = './Train_' + self.data_type + 'data/' + model_name

            name = model_name.split('_')
            model = joblib.load(model_address)
            predict = model.predict(X_red_test)
            accuracy = round(accuracy_score(y_red_test, predict) * 100, 2)
            dict_red.update({ name[0] : accuracy})
        df_red = pd.DataFrame({ 'Model' : dict_red.keys(), 'Accuracy(%)' : dict_red.values()})

        print(df_red)
        print('-----------------------------------------------------------')
        print("Model for white Mine")
        dict_white = dict()
        for model_name in list(self.model_white):
            model_address = './Train_' + self.data_type + 'data/'  + model_name

            name = model_name.split('_')
            model = joblib.load(model_address)

            model = joblib.load(model_address)
            predict = model.predict(X_white_test)
            accuracy = round(accuracy_score(y_white_test, predict) * 100, 2)
            dict_white.update({name[0]: accuracy})
        df_white = pd.DataFrame({'Model': dict_white.keys(), 'Accuracy(%)': dict_white.values()})
        print(df_white)
            #print(classification_report(predict, y_test))

if __name__ == '__main__':

    model_list_origin = [f for f in listdir('./Train_origindata') if isfile(join('./Train_origindata', f))]

    model_list_binary = [x for x in listdir('./Train_binarydata') if isfile(join('./Train_binarydata', x))]

    test_origin = Test_Model(model_list_origin, 'origin')
    test_origin.printScore()

    test_binary = Test_Model(model_list_binary, 'binary')
    test_binary.printScore()

