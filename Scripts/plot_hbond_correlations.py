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
import matplotlib.patches as mpl_patches
from sklearn.linear_model import LinearRegression
from sklearn import metrics

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

    parser.add_argument("--hbonds_file", metavar="STR", type=str,
                        help="Path to filtered H bonds file",
                        default='filter.out')

    parser.add_argument("--ic50", metavar="STR", type=str,
                        help="IC50 csv file name",
                        default='ic50.csv')

    parser.add_argument("--hbonds", nargs='*',
                        metavar="C:R[:A1, A2]", type=str, default=[],
                        help="Chain (C), residue (R) [and atoms (A1, A2)] of" +
                        "H bonds to track")

    parser.add_argument('--normalize',
                        dest='normalize',
                        action='store_true')

    parser.set_defaults(normalize=False)

    args = parser.parse_args()

    return args.traj_paths, args.hbonds_file, args.ic50, args.hbonds, \
        args.normalize


def main():
    # Parse args
    PELE_sim_paths, filtered_hbonds_path, ic50_csv, hbonds, normalize = \
        parse_args()

    if (len(hbonds) == 0):
        raise ValueError('No H bonds to track were defined')

    all_sim_it = SimIt(PELE_sim_paths)

    print(' - Simulations that will be analyzed:')
    for sim_path in all_sim_it:
        print('   - {}'.format(sim_path.name))

    data = pd.DataFrame()
    for PELE_sim_path in all_sim_it:
        print('')
        print(' - Reading data from {}'.format(PELE_sim_path))

        if (not PELE_sim_path.joinpath(filtered_hbonds_path).is_file()):
            print(' - Skipping simulation because filtered H bonds csv file ' +
                  'was missing')
            continue

        sim_data = pd.read_csv(PELE_sim_path.joinpath(filtered_hbonds_path),
                               sep=';')

        spec_hbonds = []
        for hbond in hbonds:
            for col in sim_data.columns:
                if (hbond in col and col not in spec_hbonds):
                    spec_hbonds.append(col)
                    break

        print('   - Retrieving H bonds: {}'.format(spec_hbonds))

        sim_data = sim_data.loc[:, spec_hbonds]
        sim_data['path'] = PELE_sim_path.name
        data = pd.concat((data, sim_data))

    print(' - Retreving IC50 values')
    ic50 = pd.read_csv(ic50_csv)
    data = data.merge(ic50, left_on='path', right_on='path')
    data['pIC50'] = - np.log10(data.loc[:, 'IC50'] / 1000000)

    print(' - Normalizing H bonds')
    if (normalize):
        for hbond in spec_hbonds:
            data[spec_hbonds] = data[spec_hbonds] / data['donors+acceptors']

    fig, axs = plt.subplots(int(len(spec_hbonds) / 2) +
                            len(spec_hbonds) % 2, 2, figsize=(20, 15))
    fig.suptitle('H bond frequency vs -pIC50')

    X_all = data.loc[:, spec_hbonds].values
    y_all = data['pIC50'].values

    for i, hbond in enumerate(spec_hbonds):
        ax = axs[int(i / 2)][i % 2]
        ax.set_title(hbond)
        ax.set_ylabel('Frequency')
        ax.set_xlabel('-pIC50')

        x_array = np.array([X[i] for X in X_all])
        ax.plot(y_all, x_array, ls='', c='r', marker='x')

        ax.set_axisbelow(True)
        ax.grid(True, color='white')
        ax.set_facecolor('lightgray')

        lin_reg = LinearRegression()
        print(y_all, x_array)
        lin_reg.fit(y_all.reshape(-1, 1), x_array.reshape(-1, 1))
        y_pred = lin_reg.predict(y_all.reshape(-1, 1))
        for x, xp, y, l in zip(x_array, y_pred, y_all, data['path'].values):
            if (xp == min(y_pred)):
                min_y = y
            if (xp == max(y_pred)):
                max_y = y

            ax.annotate(l.split('_')[-1],
                        (y, x),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')

        ax.plot((min_y, max_y), (min(y_pred), max(y_pred)), 'k--', linewidth=1)
        ax.autoscale(tight=False)

        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                   lw=0, alpha=0)]

        score = "r2 = {:.3f}".format(metrics.r2_score(x_array, y_pred))
        labels = []
        labels.append(score)

        ax.legend(handles, labels, loc='best', fontsize='small',
                  fancybox=True, framealpha=0.7,
                  handlelength=0, handletextpad=0)

    plt.tight_layout(h_pad=5, rect=(0, 0.05, 1, 0.95))
    plt.savefig('Hbond_correlations.png')
    plt.close()


if __name__ == "__main__":
    main()
