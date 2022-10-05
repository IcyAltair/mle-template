import configparser
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys
import traceback

from logger import Logger

SHOW_LOG = True


class MultiModel():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        self.X_train = pd.read_csv(
            self.config["SPLIT_DATA"]["X_train"], index_col=0)
        self.y_train = pd.read_csv(
            self.config["SPLIT_DATA"]["y_train"], index_col=0)
        self.X_test = pd.read_csv(
            self.config["SPLIT_DATA"]["X_test"], index_col=0)
        self.y_test = pd.read_csv(
            self.config["SPLIT_DATA"]["y_test"], index_col=0)
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
        self.project_path = os.path.join(os.getcwd(), "experiments")
        self.log_reg_path = os.path.join(self.project_path, "log_reg.sav")
        self.rand_forest_path = os.path.join(
            self.project_path, "rand_forest.sav")
        self.knn_path = os.path.join(self.project_path, "knn.sav")
        self.svm_path = os.path.join(self.project_path, "svm.sav")
        self.gnb_path = os.path.join(self.project_path, "gnb.sav")
        self.d_tree_path = os.path.join(self.project_path, "d_tree.sav")
        self.log.info("MultiModel is ready")

    def log_reg(self, predict=False) -> bool:
        classifier = LogisticRegression()
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            print(accuracy_score(self.y_test, y_pred))
        params = {'path': self.log_reg_path}
        return self.save_model(classifier, self.log_reg_path, "LOG_REG", params)

    def rand_forest(self, use_config: bool, n_trees=100, criterion="entropy", predict=False) -> bool:
        if use_config:
            try:
                classifier = RandomForestClassifier(
                    n_estimators=self.config.getint("RAND_FOREST", "n_estimators"), criterion=self.config["RAND_FOREST"]["criterion"])
            except KeyError:
                self.log.error(traceback.format_exc())
                self.log.warning(f'Using config:{use_config}, no params')
                sys.exit(1)
        else:
            classifier = RandomForestClassifier(
                n_estimators=n_trees, criterion=criterion)
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            print(accuracy_score(self.y_test, y_pred))
        params = {'n_estimators': n_trees,
                  'criterion': criterion,
                  'path': self.rand_forest_path}
        return self.save_model(classifier, self.rand_forest_path, "RAND_FOREST", params)

    def knn(self, use_config: bool, n_neighbors=5, metric="minkowski", p=2, predict=False) -> bool:
        if use_config:
            try:
                classifier = KNeighborsClassifier(n_neighbors=self.config.getint(
                    "KNN", "n_neighbors"), metric=self.config["KNN"]["metric"], p=self.config.getint("KNN", "p"))
            except KeyError:
                self.log.error(traceback.format_exc())
                self.log.warning(f'Using config:{use_config}, no params')
                sys.exit(1)
        else:
            classifier = KNeighborsClassifier(
                n_neighbors=n_neighbors, metric=metric, p=p)
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            print(accuracy_score(self.y_test, y_pred))
        params = {'n_neighbors': n_neighbors,
                  'metric': metric,
                  'p': p,
                  'path': self.knn_path}
        return self.save_model(classifier, self.knn_path, "KNN", params)

    def svm(self, use_config: bool, kernel="linear", random_state=0, predict=False) -> bool:
        if use_config:
            try:
                classifier = SVC(kernel=self.config["SVM"]["kernel"], random_state=self.config.getint(
                    "SVC", "random_state"))
            except KeyError:
                self.log.error(traceback.format_exc())
                self.log.warning(f'Using config:{use_config}, no params')
                sys.exit(1)
        else:
            classifier = SVC(kernel=kernel, random_state=random_state)
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            print(accuracy_score(self.y_test, y_pred))
        params = {'kernel': kernel,
                  'random_state': random_state,
                  'path': self.svm_path}
        return self.save_model(classifier, self.svm_path, "SVM", params)

    def gnb(self, predict=False) -> bool:
        classifier = GaussianNB()
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            print(accuracy_score(self.y_test, y_pred))
        params = {'path': self.gnb_path}
        return self.save_model(classifier, self.gnb_path, "GNB", params)

    def d_tree(self, use_config: bool, criterion="entropy", predict=False) -> bool:
        if use_config:
            try:
                classifier = RandomForestClassifier(
                    criterion=self.config["D_TREE"]["criterion"])
            except KeyError:
                self.log.error(traceback.format_exc())
                self.log.warning(f'Using config:{use_config}, no params')
                sys.exit(1)
        else:
            classifier = DecisionTreeClassifier(criterion=criterion)
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            print(accuracy_score(self.y_test, y_pred))
        params = {'criterion': criterion,
                  'path': self.d_tree_path}
        return self.save_model(classifier, self.d_tree_path, "D_TREE", params)

    def save_model(self, classifier, path: str, name: str, params: dict) -> bool:
        self.config[name] = params
        os.remove('config.ini')
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)

        self.log.info(f'{path} is saved')
        return os.path.isfile(path)


if __name__ == "__main__":
    multi_model = MultiModel()
    multi_model.log_reg(predict=True)
    multi_model.rand_forest(use_config=False, predict=True)
    multi_model.knn(use_config=False, predict=True)
    multi_model.svm(use_config=False, predict=True)
    multi_model.gnb(predict=True)
    multi_model.d_tree(use_config=False, predict=True)
