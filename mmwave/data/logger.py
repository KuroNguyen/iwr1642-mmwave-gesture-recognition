#! /usr/bin/env python

import os
import time
import pickle

import pandas as pd
from tqdm import tqdm

import colorama
from colorama import Fore

from data.gesture import GESTURE
colorama.init(autoreset=True)

class Logger:
    def __init__(self, gesture=None):
        self.logging = False
        self.gesture = gesture
        self.log_file = ''
        self.detected_time = 0
        self.empty_frames = ''
        self.frame_num = 0

    @staticmethod
    def get_data(gesture):
        if isinstance(gesture, str):
            gesture = GESTURE[gesture.upper()]
        save_dir = gesture.get_dir()
        for f in tqdm(os.listdir(save_dir), desc='Files', leave=False):
            df = pd.read_csv(os.path.join(save_dir, f))
            num_of_frames = int(df.iloc[-1]['FrameNumber'] + 1)
            sample = [[] for _ in range(num_of_frames)]

            for _, row in df.iterrows():
                if row['x'] == 'None':
                    obj = 5*[0.]
                else:
                    obj = [
                        float(row['x']),
                        float(row['y']),
                        float(row['Range']),
                        float(row['PeakValue']),
                        float(row['Velocity'])
                    ]
                sample[int(row['FrameNumber'])].append(obj)
            
            yield sample

    @staticmethod
    def get_stats(X, y):
        num_of_classes = len(set(y))
        print(f'Number of classes: {num_of_classes}')
        sample_with_max_num_of_frames = max(X, key=lambda sample: len(sample))

        max_num_of_frames = len(sample_with_max_num_of_frames)
        print(f'Maximum number of frames: {max_num_of_frames}')

        sample_with_max_num_of_objs = max(
            X, key=lambda sample: [len(frame) for frame in sample]
        )

        frame_with_max_num_of_objs = max(
            sample_with_max_num_of_objs, key=lambda obj: len(obj)
        )

        max_num_of_objs = len(frame_with_max_num_of_objs)
        print(f'Maximum num of objects: {max_num_of_objs}')

        return max_num_of_frames, max_num_of_objs, num_of_classes


    @staticmethod
    def get_all_data(refresh_data=False):
        X_file = os.path.join(os.path.dirname(__file__), '.X_data')
        y_file = os.path.join(os.path.dirname(__file__), '.y_data')
        if refresh_data:
            X = []
            y = []
            for gesture in tqdm(GESTURE, desc='Gestures'):
                for sample in Logger.get_data(gesture):
                    X.append(sample)
                    y.append(gesture.value)
            pickle.dump(X, open(X_file, 'wb'))
            pickle.dump(y, open(y_file, 'wb'))
        else:
            print('Loading cached data...', end='')
            X = pickle.load(open(X_file, 'rb'))
            y = pickle.load(open(y_file, 'rb'))
            print(f'{Fore.GREEN}Done.')
        return X, y