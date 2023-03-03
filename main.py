import os
from data.logger import Logger
from model import LstmModel

from sklearn.model_selection import train_test_split

import colorama
from colorama import Fore
colorama.init(autoreset=True)

X, y = Logger.get_all_data(refresh_data=True)
Logger.get_stats(X, y)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.3, stratify=y, random_state=12)

# model = LstmModel()
# model.train(X_train, y_train, X_val, y_val)

