import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

class DataMaker():

    def __init__(self) -> None:
        self.project_path = os.getcwd()
        self.data_path = os.path.join(self.project_path, "data", "Iris.csv")
        self.X_path = os.path.join(self.project_path, "data", "Iris_X.csv")
        self.y_path = os.path.join(self.project_path, "data", "Iris_y.csv")

    def get_data(self) -> bool:
        dataset = pd.read_csv(self.data_path)
        X = pd.DataFrame(dataset.iloc[:, 1:5].values)
        y = pd.DataFrame(dataset.iloc[:, 5:].values)
        X.to_csv(self.X_path, index=True)
        y.to_csv(self.y_path, index=True)
        return os.path.isfile(self.X_path) and os.path.isfile(self.y_path)

if __name__ == "__main__":
    data_maker = DataMaker()
    data_maker.get_data()

