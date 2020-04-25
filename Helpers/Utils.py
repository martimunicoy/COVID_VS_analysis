# -*- coding: utf-8 -*-


# External imports
import numpy as np


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


def convert_string_to_numpy_array(string):
    string = string.strip()
    string = string.strip('[]')
    return np.array(list(map(float, string.split())))
