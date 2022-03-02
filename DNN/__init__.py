
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from DataSets.ImportData import Datasets


class DeepNeuralNetworks:

    data = []

    def __init__(self, data):
        self.data = data


    def mlp_classifier(self, X_train, y_train, X_test, y_test, title):

        print("dnn classifier", title)

        mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 20), max_iter=400, solver="adam", activation="relu")
        mlp.fit(X_train, y_train)# training
        predict = mlp.predict(X_test)# prediction
        print("training rate is :{:.2f}%".format(accuracy_score(y_test, predict) * 100))
        print(classification_report(predict, y_test))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    location = "..\\DataSets\\winequality-red.csv"
    datasets = Datasets(location)
    datasets.displayLocation()
    X, y = datasets.loadData()
    datasets.getFeatures()
    #datasets.displayData()
    data = datasets.getData()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    dnn = DeepNeuralNetworks(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 25)
    dnn.mlp_classifier(X_train, y_train, X_test, y_test, "原始数据")
