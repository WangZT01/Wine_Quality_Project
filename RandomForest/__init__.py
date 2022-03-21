import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
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
        print(rf.get_params())
        return rf

    def normalization(self, X_train):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        return X_train

    def RF_GridSearch(self, X_train, y_train, X_test, y_test):
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9)
        rf = RandomForestClassifier()
        parameter_space = {
            'n_estimators' : [10, 40, 80, 100],
            'criterion' : ["gini", "entropy"],
            'max_depth' : [5, 50, 100],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        clf = GridSearchCV(rf, parameter_space, n_jobs=-1, cv=5)
        clf.fit(X_train, y_train)  # X is train samples and y is the corresponding labels
        df = pd.DataFrame(clf.cv_results_)
        print(df)
        print('Best score is: ', clf.best_score_)
        print('Best prarmeters is: ', clf.best_params_)

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
    rf.RF_GridSearch(X_train, y_train, X_test, y_test)
