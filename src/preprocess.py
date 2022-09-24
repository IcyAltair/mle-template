import os
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import traceback

from logger import Logger

TEST_SIZE = 0.2
SHOW_LOG = True


class DataMaker():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        self.project_path = os.path.join(os.getcwd(), "data")
        self.data_path = os.path.join(self.project_path, "Iris.csv")
        self.X_path = os.path.join(self.project_path, "Iris_X.csv")
        self.y_path = os.path.join(self.project_path, "Iris_y.csv")
        self.train_path = [os.path.join(self.project_path, "Train_Iris_X.csv"), os.path.join(
            self.project_path, "Train_Iris_y.csv")]
        self.test_path = [os.path.join(self.project_path, "Test_Iris_X.csv"), os.path.join(
            self.project_path, "Test_Iris_y.csv")]
        self.log.info("DataMaker is ready")

    def get_data(self) -> set:
        dataset = pd.read_csv(self.data_path)
        X = pd.DataFrame(dataset.iloc[:, 1:5].values)
        y = pd.DataFrame(dataset.iloc[:, 5:].values)
        X.to_csv(self.X_path, index=True)
        y.to_csv(self.y_path, index=True)
        if os.path.isfile(self.X_path) and os.path.isfile(self.y_path):
            self.log.info("X and y data is ready")
            return (X, y)
        else:
            self.log.error("X and y data is not ready")
            return ()

    def split_data(self, test_size=TEST_SIZE) -> set:
        try:
            X, y = self.get_data()
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)
        self.save_splitted_data(X_train, self.train_path[0])
        self.save_splitted_data(y_train, self.train_path[1])
        self.save_splitted_data(X_test, self.test_path[0])
        self.save_splitted_data(y_test, self.test_path[1])
        self.log.info("Train and test data is ready")
        return (X_train, X_test, y_train, y_test)

    def save_splitted_data(self, df: pd.DataFrame, path: str) -> bool:
        df = df.reset_index(drop=True)
        df.to_csv(path, index=True)
        self.log.info(f'{path} is saved')
        return os.path.isfile(path)


if __name__ == "__main__":
    data_maker = DataMaker()
    data_maker.split_data()
