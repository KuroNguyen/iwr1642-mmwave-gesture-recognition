import os
from mmwave.data.logger import Logger
from mmwave.data.gesture import GESTURE
from model import LstmModel

from sklearn.model_selection import train_test_split

import colorama
from colorama import Fore
colorama.init(autoreset=True)

train_ratio = 0.80
test_ratio = 0.10
validation_ratio = 0.10

X, y = Logger.get_all_data_in_range(1, 100, refresh_data=False)
Logger.get_stats(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_ratio/(train_ratio+test_ratio))

model = LstmModel()

model.train(X_train, y_train, X_valid, y_valid)
