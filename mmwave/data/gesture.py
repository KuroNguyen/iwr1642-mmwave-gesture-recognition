#! /usr/bin/env python
from enum import Enum, auto
import os

class GESTURE(Enum):
    # ARM_TO_LEFT = 0
    # ARM_TO_RIGHT = auto()
    # CLOSE_FIST_HORIZONTALLY = auto()
    # CLOSE_FIST_PERPENDICULARLY = auto()
    # HAND_AWAY = auto()
    # HAND_CLOSER = auto()
    # HAND_DOWN = auto()
    # HAND_ROTATION_PALM_DOWN = auto()
    # HAND_ROTATION_PALM_UP = auto()
    # HAND_TO_LEFT = auto()
    # HAND_TO_RIGHT = auto()
    # HAND_UP = auto()

    UP = 0
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    CW = auto()
    CCW = auto()
    Z = auto()
    S = auto()
    X = auto()

    @staticmethod
    def check(name):
        for gesture in GESTURE:
            if name.upper() == gesture.name:
                return True
        return False

    def get_dir(self):
        return os.path.join(os.path.dirname(__file__), self.name.lower())