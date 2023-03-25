#! /usr/bin/env python

import time
import glob
import os
import platform
import readline
import binascii
import serial
import pickle
import numpy as np
from copy import deepcopy

import matplotlib.pyplot as plt

from cmd import Cmd

from threading import Lock

from mmwave.utils.plotter import Plotter
from mmwave.data.logger import Logger
from mmwave.utils.util_functions import error, warning, threaded

from model import ConvModel, LstmModel, TransModel

from sklearn.model_selection import train_test_split

import colorama
from colorama import Fore

colorama.init(autoreset=True)

class ModelConsole(Cmd):
    
    def __init__(self):
        super().__init__()
        
        self.logger = Logger()
        
        self.available_model = ['conv', 'lstm', 'trans']
        self.model_type = self.available_model[0]
        self.__set_model(self.model_type)
        
        self.__set_prompt()
        print(f'{Fore.GREEN}Init done.\n')
        warning('Type \'help\' for more information.')

    def __set_prompt(self):
        self.prompt = f'{Fore.GREEN}>>{Fore.RESET} '
        
    def __set_model(self, type):
        if type == 'conv':
            self.model = ConvModel()
        elif type == 'lstm':
            self.model = LstmModel()
        elif type == 'trans':
            self.model = TransModel()
            
    def preloop(self):
        """
        Initialization before prompting user for commands.
        Despite the claims in the Cmd documentation, Cmd.preloop() is not a
        stub.
        """
        Cmd.preloop(self)  # sets up command completion
        self._hist = []  # No history yet
        self._locals = {}  # Initialize execution namespace for user
        self._globals = {}

    def postloop(self):
        """
        Take care of any unfinished business.
        Despite the claims in the Cmd documentation, Cmd.postloop() is not a
        stub.
        """
        Cmd.postloop(self)  # Clean up command completion
        print('Exiting...')

    def precmd(self, line):
        """
        This method is called after the line has been input but before
        it has been interpreted. If you want to modify the input line
        before execution (for example, variable substitution) do it here.
        """
        self._hist += [line.strip()]

        # try:
        #     info = self.plotter_queues['info'].get(False)
        #     if info == 'closed':
        #         print(f'{Fore.YELLOW}Plotter closed.\n')
        #         with self.plotting_lock:
        #             if self.plotting:
        #                 self.plotting = False
        # except queue.Empty:
        #     pass

        return line 
    
    def postcmd(self, stop, line):
        """
        If you want to stop the console, return something that evaluates to
        true. If you want to do some post command processing, do it here.
        """
        self.__set_prompt()
        return stop
    
    def emptyline(self):
        """Do nothing on empty input line"""
        pass
    
    def default(self, line):
        """
        Called on an input line when the command prefix is not recognized.
        In that case we execute the line as Python code.
        """
        try:
            exec(line) in self._locals, self._globals
        except Exception:
            error('Unknown arguments.')
            return
        
    def do_history(self, args):
        """Print a list of commands that have been entered"""
        if args != '':
            error('Unkown arguments.')
            return
        print(self._hist)

    def do_exit(self, args):
        """Exits from the console"""

        if args != '':
            error('Unkown arguments.')
            return
        
        
        os._exit(0)

    def __complete_from_list(self, complete_list, text, line):
        mline = line.partition(' ')[2]
        offs = len(mline) - len(text)
        return [s[offs:] for s in complete_list if s.startswith(mline)] 
    
    def __model_loaded(self):
        return self.model.model is not None
    
    def do_set_model(self, args=''):
        """
        Set model type used for prediction. Available models are
        \'conv\' (convolutional 1D), \'lstm\' (long short-term memory) and
        \'trans\' (transformer). Default is lstm.

        Usage:
        >> set_model conv
        >> set_model lstm
        >> set_model trans
        """
        
        if len(args.split()) > 1:
            error('Too many arguments.')
            return

        if args not in self.available_model:
            warning(f'Unknown argument: {args}')
            return

        self.model_type = args
        self.__set_model(args)
        
    def do_get_model(self, args=''):
        """
        Get current model type.

        Usage:
        >> get_model
        """
        
        if args != '':
            error('Unknown arguments.')
            return

        print(f'Current model type: {self.model_type}')
        
    def __get_file_with_name(self, name):
        print(os.path.join(name))
        return os.path.join(name)
    
    def __load_old_train_test_split_data(self, refresh=False):
        X_train_file = self.__get_file_with_name('.X_train_file')
        y_train_file = self.__get_file_with_name('.y_train_file')
        
        X_valid_file = self.__get_file_with_name('.X_valid_file')
        y_valid_file = self.__get_file_with_name('.y_valid_file')
        
        X_test_file = self.__get_file_with_name('.X_test_file')
        y_test_file = self.__get_file_with_name('.y_test_file')
        
        if (refresh):
            train_ratio = 0.80
            test_ratio = 0.10
            validation_ratio = 0.10
            
            X, y = Logger.get_all_data_in_range(1, 1000, True)
            Logger.get_stats(X, y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=9)
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_ratio/(train_ratio+test_ratio), stratify=y_train, random_state=9)
            
            pickle.dump(X_train, open(X_train_file, 'wb'))
            pickle.dump(y_train, open(y_train_file, 'wb'))
            
            pickle.dump(X_test, open(X_test_file, 'wb'))
            pickle.dump(y_test, open(y_test_file, 'wb'))
            
            pickle.dump(X_valid, open(X_valid_file, 'wb'))
            pickle.dump(y_valid, open(y_valid_file, 'wb'))
            
            return X_train, y_train, X_valid, y_valid, X_test, y_test
        else:
            X_train = pickle.load(open(X_train_file, 'rb'))   
            y_train = pickle.load(open(y_train_file, 'rb'))   
            
            X_test = pickle.load(open(X_test_file, 'rb'))   
            y_test = pickle.load(open(y_test_file, 'rb'))   
            
            X_valid = pickle.load(open(X_valid_file, 'rb'))   
            y_valid = pickle.load(open(y_valid_file, 'rb'))   
            
            return X_train, y_train, X_valid, y_valid, X_test, y_test
        
    def __load_new_train_test_split_data(self, refresh=False):
        X_train_file = self.__get_file_with_name('.new_X_train_file')
        y_train_file = self.__get_file_with_name('.new_y_train_file')
        
        X_valid_file = self.__get_file_with_name('.new_X_valid_file')
        y_valid_file = self.__get_file_with_name('.new_y_valid_file')
        
        X_test_file = self.__get_file_with_name('.new_X_test_file')
        y_test_file = self.__get_file_with_name('.new_y_test_file')
        
        if (refresh):
            train_ratio = 0.80
            test_ratio = 0.10
            validation_ratio = 0.10
            
            X, y = Logger.get_all_data_in_range(1001, 1100, True)
            Logger.get_stats(X, y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=9)
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_ratio/(train_ratio+test_ratio), stratify=y_train, random_state=9)
            
            pickle.dump(X_train, open(X_train_file, 'wb'))
            pickle.dump(y_train, open(y_train_file, 'wb'))
            
            pickle.dump(X_test, open(X_test_file, 'wb'))
            pickle.dump(y_test, open(y_test_file, 'wb'))
            
            pickle.dump(X_valid, open(X_valid_file, 'wb'))
            pickle.dump(y_valid, open(y_valid_file, 'wb'))
            
            return X_train, y_train, X_valid, y_valid, X_test, y_test
        else:
            X_train = pickle.load(open(X_train_file, 'rb'))   
            y_train = pickle.load(open(y_train_file, 'rb'))   
            
            X_test = pickle.load(open(X_test_file, 'rb'))   
            y_test = pickle.load(open(y_test_file, 'rb'))   
            
            X_valid = pickle.load(open(X_valid_file, 'rb'))   
            y_valid = pickle.load(open(y_valid_file, 'rb'))   
            
            return X_train, y_train, X_valid, y_valid, X_test, y_test
    
    def do_train(self, args=''):
        """
        Train neural network

        Command will first load cached X and y data located in
        '.X_train_file', '.y_train_file', '.X_valid_file' and '.y_valid_file' files. This data will be
        used for the training process. If you want to load new split data provide \'refresh\'.

        Usage:
        >> train
        >> train refresh
        """        
        
        if len(args.split()) > 1:
            error('Unknown arguments.')
            return

        if args == '':
            refresh_data = False
        elif args == 'refresh':
            refresh_data = True
        else:
            warning(f'Unknown argument: {args}')
            return
        
        if  not self.__model_loaded:
            error('No model has been loaded')  
        
        X_train, y_train, X_valid, y_valid, X_test, y_test = self.__load_old_train_test_split_data(refresh_data)
            
        self.model.train(X_train, y_train, X_valid, y_valid)
        
    def do_eval(self, args=''):
        """
        Evaluate neural network

        Command will first load cached first 1000 X and y data of dataset located in
        '.X_train_file', '.y_train_file', '.X_valid_file', '.y_valid_file', '.X_test_file' and '.y_test_file' files. 
        Then it will load cached 100 X and y data indexed from 1001 of the dataset located in
        '.new_X_train_file', '.new_y_train_file', '.new_X_valid_file', '.new_y_valid_file', '.new_X_test_file' and '.new_y_test_file' files.
        
        This data will beused for the evaluate process.

        Possible options: \'retrained\'
            \'retrained\': Use the retrained model to evaluate loaded data 

        Usage:
        >> eval
        >> eval retrained
        """
        
        if len(args.split()) > 1:
            error('Unknown arguments.')
            return
        
        if args == '':
            self.model.load()
        elif args == 'retrained':
            self.model.load_retrain_model()
        else:
            warning(f'Unknown argument: {args}')
            return

        old_X_train, old_y_train, old_X_valid, old_y_valid, old_X_test, old_y_test = self.__load_old_train_test_split_data()
        new_X_train, new_y_train, new_X_valid, new_y_valid, new_X_test, new_y_test = self.__load_new_train_test_split_data() 
        
        # old_X_full = np.concatenate((old_X_train, old_X_valid, old_X_test), axis=-1)
        # old_y_full = np.concatenate((old_y_train, old_y_valid, old_y_test), axis=-1)
        
        # new_X_full = np.concatenate((new_X_train, new_X_valid, new_X_test), axis=-1)
        # new_y_full = np.concatenate((new_y_train, new_y_valid, new_y_test), axis=-1)
        
        # X_full = np.concatenate((old_X_full, new_X_full), axis=-1)
        # y_full = np.concatenate((old_y_full, new_y_full), axis=-1)
        
        print(f'{Fore.GREEN} Evaluate old train dataset:')
        self.model.evaluate(old_X_train, old_y_train)
        print(f'========================')
        
        print(f'{Fore.GREEN} Evaluate old test dataset:')
        self.model.evaluate(old_X_test, old_y_test)
        print(f'========================')
        
        # print(f'{Fore.GREEN} Evaluate old full dataset:')
        # self.model.evaluate(old_X_full, old_y_full)

        print(f'{Fore.GREEN} Evaluate new train dataset:')
        self.model.evaluate(new_X_train, new_y_train)
        print(f'========================')
        
        print(f'{Fore.GREEN} Evaluate new test dataset:')
        self.model.evaluate(new_X_test, new_y_test)
        print(f'========================')
        
        # print(f'{Fore.GREEN} Evaluate new full dataset:')
        # self.model.evaluate(new_X_full, new_y_full)
        
        # print(f'{Fore.GREEN} Evaluate full dataset:')
        # self.model.evaluate(X_full, y_full)
    
@threaded
def console_thread(console):
    while True:
        console.cmdloop()
        
    
if __name__ == '__main__':
    console_thread(ModelConsole())