import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from DataSets.ImportData import Datasets
import seaborn as sns


class DeepNeuralNetworks:

    data = []

    def __init__(self, data):
        self.data = data


    def mlp_classifier(self, X_train, y_train, X_test, y_test, title):

        print("dnn classifier", title)

        mlp = MLPClassifier(hidden_layer_sizes=(100, 80, 20), max_iter=2000, solver="adam", activation="relu")
        mlp.fit(X_train, y_train)# training
        predict = mlp.predict(X_test)# prediction
        print("training rate is :{:.2f}%".format(accuracy_score(y_test, predict) * 100))
        print('score ：{:.2f}'.format(mlp.score(X_test, y_test)))
        print(classification_report(predict, y_test))

    def normalization(self, X_train):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        return X_train

    def MLP_GridSearch(self, X_train, y_train, X_test, y_test, title):
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9)
        mlp = MLPClassifier(max_iter = 100)
        parameter_space = {
            'hidden_layer_sizes' : [(100, 80, 70), (128, 64, 32),(100, 80, 20)],
            'solver' : ["adam", "sgd"],
            'activation' : ["relu", "tanh"],
            'alpha' : [0.001, 0.005, 0.0001],
            'learning_rate': ['constant', 'adaptive']
        }
        clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5)
        clf.fit(X_train, y_train)  # X is train samples and y is the corresponding labels
        df = pd.DataFrame(clf.cv_results_)
        print(df)
        print('Best score is: ', clf.best_score_)
        print('Best prarmeters is: ', clf.best_params_)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    location_red = "..\\DataSets\\winequality-red.csv"
    datasets_red = Datasets(location_red)
    datasets_red.displayLocation()
    X, y = datasets_red.loadData()
    datasets_red.getFeatures()

    data = datasets_red.getData()
    dnn = DeepNeuralNetworks(data)
    X = dnn.normalization(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 25)
    dnn.mlp_classifier(X_train, y_train, X_test, y_test, "white wine")
    dnn.MLP_GridSearch(X_train, y_train, X_test, y_test, "white wine")
