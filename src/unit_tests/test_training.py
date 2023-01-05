import configparser
import os
import unittest
import pandas as pd
import sys

# sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from src.train import MultiModel

config = configparser.ConfigParser()
config.read("config.ini")


class TestMultiModel(unittest.TestCase):

    def setUp(self) -> None:
        self.multi_model = MultiModel()

    def test_knn(self):
        self.assertEqual(self.multi_model.knn(use_config=False), True)


if __name__ == "__main__":
    unittest.main()
