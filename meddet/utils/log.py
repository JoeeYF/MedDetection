

import logging
from collections import OrderedDict
import numpy as np


def get_root_logger(log_file=None, log_level=logging.DEBUG):
    # log.info(msg) or higher will print to console and file
    # log.debug(msg) will only print to file
    
    logger = logging.getLogger('medtk')
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(log_level)

    c_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)-6s - %(message)s',
                                    datefmt="%Y-%m-%d %H:%M:%S")
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(c_formatter)
    logger.addHandler(c_handler)

    if log_file:
        f_formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)-6s - %(message)s",
                                        datefmt="%Y-%m-%d %H:%M:%S")
        f_handler = logging.FileHandler(log_file)
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(f_formatter)
        logger.addHandler(f_handler)

    return logger


class LogBuffer(object):

    def __init__(self):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output = OrderedDict()
        self.ready = False

    def clear(self):
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self):
        self.output.clear()
        self.ready = False

    def update(self, vars, count=1):
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            self.val_history[key].append(var)
            self.n_history[key].append(count)

    def average(self, n=0):
        """Average latest n values or all values"""
        assert n >= 0
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg
        self.ready = True
