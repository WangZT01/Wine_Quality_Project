
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from DataSets.ImportData import Datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class SupportVectorMachines:

    data = []

    def __init__(self, data):
        self.data = data


    def svm_classifier(self, X_train, y_train, X_test, y_test, title):

        print("svm classifier", title)

        svm = SVC()
        svm.fit(X_train, y_train)
        predict = svm.predict(X_test)# prediction
        print("training rate is :{:.2f}%".format(accuracy_score(y_test, predict) * 100))
        print('score ：{:.2f}'.format(svm.score(X_test, y_test)))
        print(classification_report(predict, y_test))
        return svm

    def normalization(self, X_train):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        return X_train

    def SVM_GridSearch(self, X_train, X_test, y_train, y_test):
        best_score = 0
        gammas = [0.001, 0.01, 0.1, 1, 10, 100]
        Cs = [0.001, 0.01, 0.1, 1, 10, 100]

        for gamma in gammas:
            for C in Cs:
                svm = SVC(kernel='rbf', gamma=gamma, C=C)
                svm.fit(X_train, y_train)

                score = svm.score(X_test, y_test)

                if score > best_score:
                    y_pred = svm.predict(X_test)
                    best_score = score
                    best_params = {'C': C, 'gamma': gamma}

        print("best score:", best_score)
        print("best params:", best_params)
        print("classification reports:\n", classification_report(y_test, y_pred))

if __name__ == '__main__':


    location_red = "..\\DataSets\\winequality-red.csv"
    datasets_red = Datasets(location_red)
    datasets_red.displayLocation()
    X, y = datasets_red.loadData()
    datasets_red.getFeatures()

    data = datasets_red.getData()
    svm = SupportVectorMachines(data)
    X = svm.normalization(X)

    pca_new = PCA(n_components=8)
    x_new = pca_new.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.35, random_state = 25)
    model = svm.svm_classifier(X_train, y_train, X_test, y_test, "red wine")
    svm.SVM_GridSearch(X_train, X_test, y_train, y_test)
