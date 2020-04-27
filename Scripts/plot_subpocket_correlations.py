# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
import os
import sys

# External imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpl_patches
from sklearn.linear_model import LinearRegression
from sklearn import metrics as skmetrics

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
                        default='ic50.csv')

    parser.add_argument("-p", "--percentile", metavar="FLOAT", type=float,
                        help="Percentile applied to the set of " +
                        "intersections that are obtained for each subpocket",
                        default=95)

    args = parser.parse_args()

    return args.traj_paths, args.subpockets, args.ic50, args.percentile


def main():
    # Parse args
    PELE_sim_paths, csv_file_name, ic50_csv, percentile = parse_args()

    all_sim_it = SimIt(PELE_sim_paths)

    print(' - Simulations that will be analyzed:')
    for sim_path in all_sim_it:
        print('   - {}'.format(sim_path.name))

    columns = []
    for PELE_sim_path in all_sim_it:
        if (not PELE_sim_path.joinpath(csv_file_name).is_file()):
            print(' - Skipping simulation because intersections csv file ' +
                  'was missing')
            continue

        data = pd.read_csv(PELE_sim_path.joinpath(csv_file_name))
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        for col in data.columns:
            if ('_intersection' in col):
                if (col not in columns):
                    columns.append(col)

    print('   - Subpockets found:')
    for col in columns:
        print('     - {}'.format(col.strip('_intersection')))

    if (len(columns) == 0):
        raise ValueError('No subpocket intersections were found in the ' +
                         'simulation paths that were supplied')

    fig, axs = plt.subplots(len(columns), 1, figsize=(20, 15))
    fig.suptitle('Subpocket-LIG volume intersection')

    for i, col in enumerate(columns):
        axs[i].set_title(col.strip('_intersection'))
        axs[i].set_ylabel('{}-LIG volume intersection ($\AA^3$)'.format(
            col.strip('_intersection')))

    subpocket_results = pd.DataFrame()
    for PELE_sim_path in all_sim_it:
        print('')
        print(' - Reading data from {}'.format(PELE_sim_path))

        if (not PELE_sim_path.joinpath(csv_file_name).is_file()):
            print(' - Skipping simulation because intersection csv file ' +
                  'is missing')
            continue

        print('   - Retrieving subpocket intersections')
        data = pd.read_csv(PELE_sim_path.joinpath(csv_file_name))

        metrics = [PELE_sim_path.name, ]
        for col in columns:
            values = data[col].values
            metrics.append(np.percentile(values, percentile))

        subpocket_results = pd.concat([subpocket_results,
                                      pd.DataFrame([metrics],
                                                   columns=['path', ] +
                                                   columns)])

    print(' - Retrieving IC50 values')
    ic50 = pd.read_csv(ic50_csv)
    subpocket_results = subpocket_results.merge(
        ic50, left_on='path', right_on='path')
    subpocket_results['pIC50'] = - np.log10(
        subpocket_results.loc[:, 'IC50'] / 1000000)

    fig, axs = plt.subplots(int(len(columns) / 2) + len(columns) % 2, 2,
                            figsize=(15, 5 * int(len(columns) / 2 +
                                     len(columns) % 2)))
    fig.suptitle('Subpocket occupancy vs -pIC50')

    X_all = subpocket_results.loc[:, columns].values
    y_all = subpocket_results['pIC50'].values

    for i, col in enumerate(columns):
        ax = axs[int(i / 2)][i % 2]
        ax.set_title(col.strip('_intersection'))
        ax.set_ylabel('{}-percentile of {} occupancies'.format(
            percentile, col.strip('_intersection')))
        ax.set_xlabel('-pIC50')

        x_array = np.array([X[i] for X in X_all])
        ax.plot(y_all, x_array, ls='', c='r', marker='x')

        ax.set_axisbelow(True)
        ax.grid(True, color='white')
        ax.set_facecolor('lightgray')

        lin_reg = LinearRegression()
        lin_reg.fit(y_all.reshape(-1, 1), x_array.reshape(-1, 1))
        y_pred = lin_reg.predict(y_all.reshape(-1, 1))
        for x, xp, y, path in zip(x_array, y_pred, y_all,
                                  subpocket_results['path'].values):
            if (xp == min(y_pred)):
                min_y = y
            if (xp == max(y_pred)):
                max_y = y

            ax.annotate(path,
                        (y, x),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')

        ax.plot((min_y, max_y), (min(y_pred), max(y_pred)), 'k--', linewidth=1)
        ax.autoscale(tight=False)

        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                   lw=0, alpha=0)]

        score = "r2 = {:.3f}".format(skmetrics.r2_score(x_array, y_pred))
        labels = []
        labels.append(score)

        ax.legend(handles, labels, loc='best', fontsize='small',
                  fancybox=True, framealpha=0.7,
                  handlelength=0, handletextpad=0)

    # Empty unpaired axis
    if (i % 2 == 0):
            fig.delaxes(axs[int(i / 2)][1])

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig('subpocket_correlations.png')
    plt.close()


if __name__ == "__main__":
    main()
