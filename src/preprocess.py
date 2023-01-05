import configparser
import os
import pandas as pd
import sys
import traceback

from logger import Logger

SHOW_LOG = True


class DataMaker():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.project_path = os.path.join(os.getcwd(), "data")
        self.train_path = os.path.join(self.project_path, "fashion-mnist_train.csv")
        self.test_path = os.path.join(self.project_path, "fashion-mnist_test.csv")
        self.X_train_path = os.path.join(self.project_path, "fashion_X_train.csv")
        self.y_train_path = os.path.join(self.project_path, "fashion_y_train.csv")
        self.X_test_path = os.path.join(self.project_path, "fashion_X_test.csv")
        self.y_test_path = os.path.join(self.project_path, "fashion_y_test.csv")
        self.data_path = [os.path.join(self.train_path, 'fashion-mnist_train.csv'),
                          os.path.join(self.test_path, 'fashion-mnist_test.csv')]
        self.log.info("DataMaker is ready")

    def get_data(self) -> bool:


        if os.path.isfile(self.test_path) and os.path.isfile(self.train_path):
            self.log.info("test and train data is ready")
            self.config["DATA"] = {'train_data': self.train_path,
                                   'test_data': self.test_path}
            return os.path.isfile(self.test_path) and os.path.isfile(self.train_path)
        else:
            self.log.error("train and test data is not ready")
            return False


    def split_data(self):

        try:
           self.get_data()

        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)


        test = pd.read_csv(self.test_path)
        train = pd.read_csv(self.train_path)
        X_train = pd.DataFrame(train.iloc[:, 1:].values)
        y_train = pd.DataFrame(train.iloc[:, 0].values)
        X_test = pd.DataFrame(test.iloc[:, 1:].values)
        y_test = pd.DataFrame(test.iloc[:, 0].values)

        X_train.to_csv(self.X_train_path, index=True)
        y_train.to_csv(self.y_train_path, index=True)
        X_test.to_csv(self.X_test_path, index=True)
        y_test.to_csv(self.y_test_path, index=True)


        self.save_splitted_data(X_train, self.X_train_path)
        self.save_splitted_data(y_train, self.y_train_path)
        self.save_splitted_data(X_test, self.X_test_path)
        self.save_splitted_data(y_test, self.y_test_path)
        self.config["SPLIT_DATA"] = {'X_train': self.X_train_path,
                                     'y_train': self.y_train_path,
                                     'X_test': self.X_test_path,
                                     'y_test': self.y_test_path}

        self.log.info("X_train, y_train and X_test, y_test data is ready")

        with open('../config.ini', 'w') as configfile:
            self.config.write(configfile)
        return os.path.isfile(self.X_train_path) and\
            os.path.isfile(self.y_train_path) and\
            os.path.isfile(self.X_test_path) and \
            os.path.isfile(self.y_test_path)

    def save_splitted_data(self, df: pd.DataFrame, path: str) -> bool:
        df = df.reset_index(drop=True)
        df.to_csv(path, index=True)
        self.log.info(f'{path} is saved')
        return os.path.isfile(path)


if __name__ == "__main__":
    data_maker = DataMaker()
    data_maker.split_data()
