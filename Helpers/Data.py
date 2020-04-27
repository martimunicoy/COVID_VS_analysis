# -*- coding: utf-8 -*-


# External imports
import pandas as pd
import numpy as np

# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


def build_dataframe_from_results(results, columns):
    data = pd.concat(
        [pd.DataFrame([r], columns=columns) for r in np.concatenate(results)])

    return data
