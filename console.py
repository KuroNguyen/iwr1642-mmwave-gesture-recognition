#! /usr/bin/env python

import time
import glob
import os
import platform
import readline
import binascii
import serial
from copy import deepcopy

import matplotlib.pyplot as plt

from cmd import Cmd

from threading import Lock
import queue
from queue import Queue

from communication.connection import Connection, mmWave
from communication.parser import Parser
from data.formats import Formats
from data.gesture import GESTURE
from data.logger import Logger

import colorama
from colorama import Fore
from utils.flasher import Flasher

from utils.handlers import SignalHandler
from utils.util_functions import warning
colorama.init(autoreset=True)

class Console(Cmd):
    def __init__(self, plotter_queues):
        super().__init__()

        # Connection
        self.cli_port = None
        self.cli_rate = None
        self.data_port = None
        self.data_rate = None

        self.default_cli_rate = 115200
        self.default_data_rate = 921600

        self.firmware_dir = 'firmware/'
        self.flasher = None

        self.__mmwave_init()
        if self.mmwave is None or self.mmwave.connected() is False:
            print('Try connecting manually. Type \'help connect\' for more info.\n')

        # Configuration
        self.config_dir = 'communication/profiles/'
        self.configured = False
        self.default_config = 'profile'
        self.config = None

        self.logger = Logger()

        self.model_type = 'lstm'
        self.__set_model(self.model_type)

        # Catching signals
        self.console_queue = Queue()
        SignalHandler(self.console_queue)

        # Threading stuff
        self.listening_lock = Lock()
        self.printing_lock = Lock()
        self.plotting_lock = Lock()
        self.predicting_lock = Lock()
        self.logging_lock = Lock()

        self.listening = False
        self.printing = False
        self.plotting = False
        self.predicting = False
        self.logging = False

        self.logging_queue = Queue()
        self.data_queue = Queue()
        self.model_queue = Queue()
        self.plotter_queues = plotter_queues

        self.__set_prompt()
        print(f'{Fore.GREEN}Init done.\n')
        print(f'{Fore.MAGENTA}--- mmWave console ---')
        warning('Type \'help\' for more information.')

    def __mmwave_init(self):
        self.mmwave = None
        if self.cli_port is None or self.data_port is None:
            print('Looking for ports...', end='')  
            ports = mmWave.find_ports()

            if len(ports) < 2:
                print(f'{Fore.RED}Ports not found!')
                print(f'{Fore.YELLOW}Auto-detection is only applicable for',
                      f'{Fore.YELLOW}eval boards with XDS110.')
                return
            
            if len(ports) > 2:
                print(f'{Fore.YELLOW}Multiple ports detected.',
                      f'{Fore.YELLOW}Selecting ports {ports[0]} and {ports[1]}.')
                ports = ports[:2]

            if platform.system() == 'Windows':
                ports.sort(reverse=True)

            self.cli_port = ports[0]
            self.data_port = ports[1]

        self.mmwave = mmWave(self.cli_port, self.data_port,
                             cli_rate=self.default_cli_rate,
                             data_rate=self.default_data_rate)
        self.mmwave.connect()
        if self.mmwave.connected():
            self.flasher = Flasher(self.mmwave.cli_port)

        
