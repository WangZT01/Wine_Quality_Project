from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from DataSets.ImportData import Datasets
from sklearn.decomposition import PCA

class RandomForest:

    data = []

    def __init__(self, data):
        self.data = data

    def rf_classifier(self, X_train, y_train, X_test, y_test, title):

        print("Random Forest classifier", title)

        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        predict = rf.predict(X_test)# prediction
        print("training rate is :{:.2f}%".format(accuracy_score(y_test, predict) * 100))
        print('score ：{:.2f}'.format(rf.score(X_test, y_test)))
        print(classification_report(predict, y_test))
        return rf

    def normalization(self, X_train):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        return X_train


if __name__ == '__main__':


    location_red = "..\\DataSets\\winequality-red.csv"
    datasets_red = Datasets(location_red)
    datasets_red.displayLocation()
    X, y = datasets_red.loadData()
    datasets_red.getFeatures()

    data = datasets_red.getData()
    rf = RandomForest(data)
    X = rf.normalization(X)

    pca_new = PCA(n_components=8)
    x_new = pca_new.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.35, random_state = 25)
    model = rf.rf_classifier(X_train, y_train, X_test, y_test, "red wine")

