# -*- coding: utf-8 -*-


# Standard imports
from pathlib import Path
from typing import Tuple

# External imports
import pandas as pd
from ast import literal_eval


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


def extract_hbonds(hbonds_path: Path) -> Tuple[pd.DataFrame, int, int]:
    output_info_path = Path(str(hbonds_path).replace(hbonds_path.suffix, '')
                            + '.info')

    if (output_info_path.is_file()):
        with open(str(output_info_path), 'r') as file:
            # Skip four header lines
            file.readline()
            n_donors = int(file.readline().split()[0])
            n_acceptors = int(file.readline().split()[0])
            file.readline()
    else:
        print('   - Warning: H bonds info output file not found')
        n_donors = -1
        n_acceptors = -1

    data = pd.read_csv(str(hbonds_path))
    data.hbonds = data.hbonds.apply(literal_eval)

    return data, n_donors, n_acceptors
