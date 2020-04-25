# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
from pathlib import Path
import os

# External imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# PELE imports
import sys
sys.path.append('..')
from Helpers.Utils import convert_string_to_numpy_array


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("csv_file", metavar="PATH", type=str,
                        help="Path to subpockets csv file")

    parser.add_argument("-o", "--output_path",
                        metavar="PATH", type=str,
                        default="subpocket_centroids_analysis")

    args = parser.parse_args()

    return args.csv_file, args.output_path


def main():
    # Parse args
    csv_file, out_path = parse_args()

    csv_file = Path(csv_file)
    out_path = Path(out_path)

    if (not csv_file.is_file()):
        raise ValueError('Wrong path to csv file')

    if (not out_path.parent.is_dir()):
        raise ValueError('Wrong output path')

    if (not out_path.is_dir()):
        os.mkdir(str(out_path))

    data = pd.read_csv(csv_file)

    subpockets = []
    for column in data.columns:
        if ('centroid' in column):
            subpockets.append(column.strip('_centroid'))
            data[column] = data[column].apply(convert_string_to_numpy_array)

    if (len(subpockets) == 0):
        raise RuntimeError('No subpocket was found at {}'.format(csv_file))
    else:
        print(' - The following subpocket{} found:'.format(
            [' was', 's were'][len(subpockets) > 1]))
        for subpocket in subpockets:
            print('   - {}'.format(subpocket))

    data['simulation'].fillna('.', inplace=True)
    simulations = set(data['simulation'].to_list())

    print(' - {} simulation{} found'.format(
        len(simulations), [' was', 's were'][len(simulations) > 1]))

    for subpocket in subpockets:
        for simulation in simulations:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            fig.suptitle(
                'Subpocket centroids for simulation {}'.format(simulation))

            ax1.set_title('X')
            ax2.set_title('Y')
            ax3.set_title('Z')

            simulation_data = data.loc[data['simulation'] == simulation]

            centroid_coords = np.array(
                simulation_data['{}_centroid'.format(subpocket)].to_list())

            ax1.boxplot(centroid_coords[:, 0])
            ax2.boxplot(centroid_coords[:, 1])
            ax3.boxplot(centroid_coords[:, 2])

            plt.tight_layout(rect=(0, 0, 0.95, 0.95))
            plt.savefig(out_path.joinpath('{}_centroid_analysis.png'.format(
                simulation.replace('.', 'simulation'))))
            plt.close()


if __name__ == "__main__":
    main()
