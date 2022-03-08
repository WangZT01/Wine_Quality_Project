from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
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

        mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 20), max_iter=2000, solver="adam", activation="relu")
        mlp.fit(X_train, y_train)# training
        predict = mlp.predict(X_test)# prediction
        print("training rate is :{:.2f}%".format(accuracy_score(y_test, predict) * 100))
        print('score ：{:.2f}'.format(mlp.score(X_test, y_test)))
        print(classification_report(predict, y_test))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    '''
    location_red = "..\\DataSets\\winequality-red.csv"
    datasets_red = Datasets(location_red)
    datasets_red.displayLocation()
    X, y = datasets_red.loadData()
    datasets_red.getFeatures()
    #datasets_red.displayData()
    '''
    location_white = "..\\DataSets\\winequality-white.csv"
    datasets_white = Datasets(location_white)
    datasets_white.displayLocation()
    X2, y2 = datasets_white.loadData()
    datasets_white.getFeatures()
    datasets_white.displayData()

    pca = PCA(n_components=8, whiten=True).fit(X2)
    X_pca = pca.transform(X2)

    data = datasets_white.getData()
    scaler = StandardScaler()
    X2 = scaler.fit_transform(X2)
    dnn = DeepNeuralNetworks(data)
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size = 0.35, random_state = 25)
    dnn.mlp_classifier(X_train, y_train, X_test, y_test, "white wine")
