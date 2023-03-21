import os
from mmwave.data.logger import Logger
from mmwave.data.gesture import GESTURE
from model import LstmModel
import pickle

from sklearn.model_selection import train_test_split

import colorama
from colorama import Fore
colorama.init(autoreset=True)

train_ratio = 0.80
test_ratio = 0.10
validation_ratio = 0.10

def get_file_with_name(name):
    return os.path.join(os.path.dirname(__file__), name)

def do_train(refresh=False):
    
    X_train_file = get_file_with_name('.X_train_file')
    y_train_file = get_file_with_name('.y_train_file')
    
    X_test_file = get_file_with_name('.X_test_file')
    y_test_file = get_file_with_name('.y_test_file')
    
    X_valid_file = get_file_with_name('.X_valid_file')
    y_valid_file = get_file_with_name('.y_valid_file')
    
    if (refresh):
        X, y = Logger.get_all_data_in_range(1, 1000, refresh_data=True)
        Logger.get_stats(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=9)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_ratio/(train_ratio+test_ratio), stratify=y_train, random_state=9)

        pickle.dump(X_train, open(X_train_file, 'wb'))
        pickle.dump(y_train, open(y_train_file, 'wb'))
        
        pickle.dump(X_test, open(X_test_file, 'wb'))
        pickle.dump(y_test, open(y_test_file, 'wb'))
        
        pickle.dump(X_valid, open(X_valid_file, 'wb'))
        pickle.dump(y_valid, open(y_valid_file, 'wb'))

    else:
        X_train = pickle.load(open(X_train_file, 'rb'))   
        y_train = pickle.load(open(y_train_file, 'rb'))   
        
        X_test = pickle.load(open(X_test_file, 'rb'))   
        y_test = pickle.load(open(y_test_file, 'rb'))   
        
        X_valid = pickle.load(open(X_valid_file, 'rb'))   
        y_valid = pickle.load(open(y_valid_file, 'rb'))   
        
    model = LstmModel()
    model.train(X_train, y_train, X_valid, y_valid)
    
    
def do_evaluate():
    
    X_test_file = get_file_with_name('.X_test_file')
    y_test_file = get_file_with_name('.y_test_file')
    
    X_test = pickle.load(open(X_test_file, 'rb'))   
    y_test = pickle.load(open(y_test_file, 'rb'))   
    
    print(f'{Fore.YELLOW} test size: {len(X_test)}')
        
    model = LstmModel()
    model.load()
    
    model.evaluate(X_test, y_test)
    
def do_evaluate_with_new_data():
    
    X, y = Logger.get_all_data_in_range(1001, 1100, refresh_data=True)
    Logger.get_stats(X, y)
    
    print(f'{Fore.YELLOW} test size: {len(X)}')
    
    model = LstmModel()
    model.load()
    
    model.evaluate(X, y)

def do_retrain(refresh=False):
    X_train_file = get_file_with_name('.new_X_train_file')
    y_train_file = get_file_with_name('.new_y_train_file')
    
    X_test_file = get_file_with_name('.new_X_test_file')
    y_test_file = get_file_with_name('.new_y_test_file')
    
    X_valid_file = get_file_with_name('.new_X_valid_file')
    y_valid_file = get_file_with_name('.new_y_valid_file')
    
    if (refresh):
        X, y = Logger.get_all_data_in_range(1001, 1100, refresh_data=True)
        Logger.get_stats(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=9)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_ratio/(train_ratio+test_ratio), stratify=y_train, random_state=9)

        pickle.dump(X_train, open(X_train_file, 'wb'))
        pickle.dump(y_train, open(y_train_file, 'wb'))
        
        pickle.dump(X_test, open(X_test_file, 'wb'))
        pickle.dump(y_test, open(y_test_file, 'wb'))
        
        pickle.dump(X_valid, open(X_valid_file, 'wb'))
        pickle.dump(y_valid, open(y_valid_file, 'wb'))

    else:
        X_train = pickle.load(open(X_train_file, 'rb'))   
        y_train = pickle.load(open(y_train_file, 'rb'))   
        
        X_test = pickle.load(open(X_test_file, 'rb'))   
        y_test = pickle.load(open(y_test_file, 'rb'))   
        
        X_valid = pickle.load(open(X_valid_file, 'rb'))   
        y_valid = pickle.load(open(y_valid_file, 'rb'))   
        
    model = LstmModel()
    model.load()
    model.retrain(X_train, y_train, X_valid, y_valid)

# do_retrain(refresh=True)

def do_evaluate_new_model():
    X_test_file = get_file_with_name('.new_X_test_file')
    y_test_file = get_file_with_name('.new_y_test_file')
    
    X_test = pickle.load(open(X_test_file, 'rb'))   
    y_test = pickle.load(open(y_test_file, 'rb'))   
    
    print(f'{Fore.YELLOW} test size: {len(X_test)}')
        
    model = LstmModel()
    model.load_retrain_model()
    
    model.evaluate(X_test, y_test)
    
    # X_test_file = get_file_with_name('.X_test_file')
    # y_test_file = get_file_with_name('.y_test_file')
    
    # X_test = pickle.load(open(X_test_file, 'rb'))   
    # y_test = pickle.load(open(y_test_file, 'rb')) 
    
    # model = LstmModel()
    # model.load_retrain_model()
    
    # model.evaluate(X_test, y_test)
    
do_evaluate_new_model()