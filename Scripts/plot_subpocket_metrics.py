# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
import os
import sys
from collections import defaultdict

# External imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Local imports
SCRIPT_PATH = os.path.dirname(__file__)
sys.path.append(os.path.abspath(SCRIPT_PATH + '/..'))
from Helpers.PELEIterator import SimIt


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

    parser.add_argument("--subpockets", metavar="STR", type=str,
                        help="Subpockets csv file name",
                        default='subpockets.csv')

    parser.add_argument("--ic50", metavar="STR", type=str,
                        help="IC50 csv file name",
                        default=None)

    parser.add_argument('-m', '--metric', default='intersection',
                        nargs='?', type=str,
                        choices=['intersection', 'nonpolar_intersection',
                                 'positive_charge', 'negative_charge'],
                        help='Metric whose distribution will be plotted')

    args = parser.parse_args()

    return args.traj_paths, args.subpockets, args.ic50, args.metric


def main():
    # Parse args
    PELE_sim_paths, csv_file_name, ic50_csv, metric = parse_args()

    all_sim_it = SimIt(PELE_sim_paths)

    print(' - Simulations that will be analyzed:')
    for sim_path in all_sim_it:
        print('   - {}'.format(sim_path.name))

    columns = []
    for PELE_sim_path in all_sim_it:
        if (not PELE_sim_path.joinpath(csv_file_name).is_file()):
            print(' - Skipping simulation because intersections csv file '
                  + 'was missing')
            continue

        data = pd.read_csv(PELE_sim_path.joinpath(csv_file_name))
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        for col in data.columns:
            if (col.endswith(metric)):
                if (metric == 'intersection'
                        and col.endswith('nonpolar_intersection')):
                    continue
                if (col not in columns):
                    columns.append(col)

    pretty_metric = metric.replace('_', ' ')
    if metric == 'intersection' or metric == 'nonpolar_intersection':
        pretty_metric = 'volume ' + pretty_metric
        units = '$\AA^3$'
    else:
        units = 'a.u.'

    print('   - Subpockets found:')
    for col in columns:
        print('     - {}'.format(col.strip('_' + metric)))

    if (len(columns) == 0):
        raise ValueError('No subpocket {} '.format(pretty_metric)
                         + 'were found in the simulation paths that '
                         + 'were supplied')

    fig, axs = plt.subplots(len(columns), 1, figsize=(20, 15))
    fig.suptitle('Subpocket-LIG {}'.format(pretty_metric))

    for i, col in enumerate(columns):
        axs[i].set_title(col.strip('_' + metric))
        axs[i].set_ylabel('{}-LIG {} ({})'.format(
            pretty_metric, col.strip('_' + metric), units))

    metrics = defaultdict(dict)
    for PELE_sim_path in all_sim_it:

        if (not PELE_sim_path.joinpath(csv_file_name).is_file()):
            print(' - Skipping simulation because '
                  + '{} csv file is missing'.format(pretty_metric))
            continue

        data = pd.read_csv(PELE_sim_path.joinpath(csv_file_name))

        for i, col in enumerate(columns):
            metrics[col][PELE_sim_path.name] = data[col].to_numpy()

    try:
        if (ic50_csv is None):
            raise ValueError
        ic50_data = pd.read_csv(ic50_csv)

        ic50_data['-pIC50'] = - np.log10(ic50_data.loc[:, 'IC50'] / 1000000)
        ic50_data.sort_values(by='-pIC50', inplace=True)

        for i, col in enumerate(columns):
            ordered_intersects = []
            ordered_labels = []
            for path, pic50 in zip(ic50_data['path'], ic50_data['-pIC50']):
                if (path in metrics[col]):
                    ordered_intersects.append(metrics[col][path])
                    ordered_labels.append(path)

            axs[i].boxplot(ordered_intersects,
                           labels=ordered_labels,
                           showfliers=False)

    except ValueError:
        print(' - Unordered plot')
        for i, col in enumerate(columns):
            axs[i].boxplot(list(metrics[col].values()),
                           labels=[i.split('_')[1] for i in
                                   list(metrics[col].keys())],
                           showfliers=False)

    for ax in axs:
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)

    plt.tight_layout(h_pad=5, rect=(0, 0.05, 1, 0.95))
    plt.savefig('subpocket_{}.png'.format(metric))
    plt.close()


if __name__ == "__main__":
    main()
