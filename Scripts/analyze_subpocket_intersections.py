# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
from pathlib import Path
import os
import sys
from collections import defaultdict

# External imports
import pandas as pd
import numpy as np
from multiprocessing import Pool

# Local imports
SCRIPT_PATH = os.path.dirname(__file__)
sys.path.append(os.path.abspath(SCRIPT_PATH + '/..'))
from Helpers.Utils import convert_string_to_numpy_array
from Helpers.PELEIterator import SimIt
from Helpers.ReportUtils import extract_metrics


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


def parse_args():
    parser = ap.ArgumentParser()

    parser.add_argument("traj_paths", metavar="PATH", type=str,
                        nargs='*',
                        help="Path to PELE trajectory files")

    parser.add_argument("--csv_name",
                        metavar="PATH", type=str,
                        default="subpockets.csv",
                        help='Name of the subpockets csv file')

    parser.add_argument("-o", "--output_name",
                        metavar="PATH", type=str,
                        default="intersections_info.txt",
                        help='Intersections output file name')

    args = parser.parse_args()

    return args.traj_paths, args.csv_name, args.output_name


def main():
    # Parse args
    PELE_sim_paths, csv_file, output_name = parse_args()

    all_sim_it = SimIt(PELE_sim_paths)

    print(' - Simulations that will be analyzed:')
    for sim_path in all_sim_it:
        print('   - {}'.format(sim_path.name))

    for PELE_sim_path in all_sim_it:
        print('')
        print(' - Analyzing {}'.format(PELE_sim_path))

        csv_path = PELE_sim_path.joinpath(csv_file)

        if (not csv_path.is_file()):
            print(' - Skipping simulation because intersections csv file ' +
                  'was missing')
            continue

        data = pd.read_csv(str(csv_path))
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        columns = []
        print('   - Subpockets found:')
        for col in data.columns:
            if ('_intersection' in col):
                columns.append(col)
                print('     - {}'.format(col.strip('_intersection')))

        if (len(columns) == 0):
            print(' - Skipping simulation because no subpocket was found')
            continue

        with open(str(PELE_sim_path.joinpath(output_name)), 'w') as f:
            f.write('   - Subpocket results:\n')
            for col in columns:
                intersects = data.loc[:, col].to_numpy()
                f.write('   - {}:\n'.format(col.strip('_intersection')))
                f.write('     - Mean: {: 7.2f}\n'.format(np.mean(intersects)))
                f.write('     - Min: {: 7.2f}\n'.format(np.min(intersects)))
                f.write('     - 5th percentile: {: 7.2f}\n'.format(
                    np.percentile(intersects, 5)))
                f.write('     - 1st quartile: {: 7.2f}\n'.format(
                    np.percentile(intersects, 25)))
                f.write('     - Median: {: 7.2f}\n'.format(
                    np.median(intersects)))
                f.write('     - 3rd quartile: {: 7.2f}\n'.format(
                    np.percentile(intersects, 75)))
                f.write('     - 95th percentile: {: 7.2f}\n'.format(
                    np.percentile(intersects, 95)))
                f.write('     - Max: {: 7.2f}\n'.format(np.max(intersects)))


if __name__ == "__main__":
    main()
