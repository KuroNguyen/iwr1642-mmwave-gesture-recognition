#! /usr/bin/env python

import os
from abc import ABC, abstractmethod

import numpy as np

import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import callbacks

import colorama
from colorama import Fore
colorama.init(autoreset=True)

class Model(ABC):
    def __init__(self, num_of_frames=50, num_of_objs=65, num_of_data_in_obj=5,
                 num_of_classes=9):
        self.model = None
        self.model_file = os.path.join(os.path.dirname(__file__), '.model')
        print(f'{Fore.RED}Model file: {self.model_file}.')

        self.num_of_frames = num_of_frames
        self.num_of_objs = num_of_objs
        self.num_of_data_in_obj = num_of_data_in_obj
        self.num_of_classes = num_of_classes

        self.frame_size = self.num_of_objs*self.num_of_data_in_obj

    def __padd_data(self, data):
        padded_data = []

        print(f'{Fore.GREEN} Data shape: {len(data)}')

        # Pad objects
        zero_obj = [0.]*self.num_of_data_in_obj
        for sample in data:
            if len(sample) > self.num_of_objs:
                sample = sample[:self.num_of_objs]

            padded_sample = preprocessing.sequence.pad_sequences(
                sample, maxlen=self.num_of_objs, dtype='float32',
                padding='post', value=zero_obj)

            padded_data.append(padded_sample)

        # Pad frames
        zero_frame = [zero_obj for _ in range(self.num_of_objs)]
        padded_data = preprocessing.sequence.pad_sequences(
            padded_data, maxlen=self.num_of_frames, dtype='float32',
            padding='post', value=zero_frame)

        return np.asarray(padded_data)

    def __prep_data(self, X, y=None):
        # X has been already normalized while importing the data from .csv files
        X = self.__padd_data(X)
        X = X.reshape((len(X), self.num_of_frames, self.frame_size))

        if y is not None:
            y = utils.to_categorical(y)
            return X, y

        return X

    @abstractmethod
    def create_model(self):
        pass

    def train(self, X_train, y_train, X_val, y_val):
        X_train, y_train = self.__prep_data(X_train, y_train)
        X_val, y_val = self.__prep_data(X_val, y_val)

        self.create_model()
        self.model.summary()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        self.model.fit(X_train, y_train, epochs=1000,
                       validation_data=(X_val, y_val),
                       callbacks=[callbacks.EarlyStopping(patience=100,
                                                          restore_best_weights=True),
                                  callbacks.ModelCheckpoint(self.model_file,
                                                            verbose=True,
                                                            save_best_only=True)])

    def load(self):
        print('Loading model...', end='')
        self.model = tf.keras.models.load_model(self.model_file)
        print(f'{Fore.GREEN}Done.')

    def evaluate(self, X, y):
        if self.model is None:
            print(f'{Fore.RED}Model not created.')
            return

        X, y = self.__prep_data(X, y)
        preds = self.model.evaluate(X, y)
        print(f'Loss: {round(preds[0], 4)}', end=' ')
        print(f'Acc: {round(preds[1], 4)}')

    def predict(self, X, debug=False):
        if self.model is None:
            print(f'{Fore.RED}Model not created.')
            return

        print(self.model)
        y_pred = self.model.predict(self.__prep_data(X))
        best_guess = [y_pred[0].tolist().index(x) for x in sorted(y_pred[0], reverse=True)]
        best_value = sorted(y_pred[0], reverse=True)

        if debug:
            for guess, val in zip(best_guess, best_value):
                print(f'{Fore.YELLOW}Best guess: {GESTURE(guess).name.lower()}: {val:.2f}')
            print(f'{Fore.CYAN}------------------------------\n')

        if best_value[0] >= .9:
            print(f'{Fore.GREEN}Gesture recognized:',
                  f'{Fore.BLUE}{GESTURE(best_guess[0]).name.lower()}')
            print(f'{Fore.CYAN}==============================\n')

class LstmModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_file = os.path.join(os.path.dirname(__file__), '.lstm_model')

    def create_model(self):
        self.model = tf.keras.Sequential([
            layers.InputLayer(input_shape=(self.num_of_frames, self.frame_size)),
            layers.LSTM(256, recurrent_dropout=.5, dropout=.5, return_sequences=True),
            layers.LSTM(256, recurrent_dropout=.5, dropout=.5, return_sequences=True),

            layers.GlobalAveragePooling1D(),

            layers.Dense(128),
            layers.PReLU(),
            layers.Dropout(.5),

            layers.Dense(self.num_of_classes, activation='softmax')
        ])