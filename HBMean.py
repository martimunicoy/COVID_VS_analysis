# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
import glob
from collections import defaultdict
from pathlib import Path

# External imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("hbonds_data_paths", metavar="PATH", type=str,
                        nargs='*',
                        help="Path to H bonds data files")
    parser.add_argument("--epochs_to_ignore", nargs='*',
                        metavar="N", type=int, default=[],
                        help="PELE epochs whose H bonds will be ignored")
    parser.add_argument("--trajectories_to_ignore", nargs='*',
                        metavar="N", type=int, default=[],
                        help="PELE trajectories whose H bonds will be ignored")
    parser.add_argument("--models_to_ignore", nargs='*',
                        metavar="N", type=int, default=[],
                        help="PELE models whose H bonds will be ignored")
    parser.add_argument("--histogram_path",
                        metavar="PATH", type=str, default=None,
                        help="Output path to save the histrogram")

    args = parser.parse_args()

    return args.hbonds_data_paths, args.epochs_to_ignore, \
        args.trajectories_to_ignore, args.models_to_ignore, args.histogram_path


def create_df(hb_path):
    rows_df = []
    with open(hb_path) as file:
        rows = file.readlines()
        for row in rows[2:]:
            rows_df.append(row.split())
    return rows_df


def get_n_hbonds_from_df(df, hb_path, epochs_to_ignore,
                         trajectories_to_ignore, models_to_ignore):
    n_hbonds = []

    for row in df:
        try:
            epoch, trajectory, model = map(int, row[0:3])
        except (IndexError, ValueError):
            print ("get_hbond_atoms_from_df Warning: found row with non" +
                   " valid format at {}:".format(hb_path))
            print(" {}".format(row))

        if ((epoch in epochs_to_ignore) or
            (trajectory in trajectories_to_ignore) or
                (model in models_to_ignore)):
            continue

        try:
            residues = row[3].split(',')
            n_hbonds.append(len(residues))

        except IndexError:
            n_hbonds.append(0)

    return n_hbonds


def main():
    hb_paths, epochs_to_ignore, trajectories_to_ignore, \
        models_to_ignore, histogram_path = parse_args()

    hb_paths_list = []
    if (type(hb_paths) == list):
        for hb_path in hb_paths:
            hb_paths_list += glob.glob(hb_path)
    else:
        hb_paths_list = glob.glob(hb_paths)

    print('Average of H bonds:')

    total_n_hbonds = []
    for hb_path in hb_paths_list:
        df = create_df(hb_path)
        n_hbonds = get_n_hbonds_from_df(df, hb_path, epochs_to_ignore,
                                        trajectories_to_ignore,
                                        models_to_ignore)
        total_n_hbonds += n_hbonds
        if (len(n_hbonds) != 0):
            print(' - {:100}: {:6.1f}'.format(str(Path(hb_path).parent),
                                              np.mean(n_hbonds)))

    print('Average among all simulations: {:6.1f}'.format(
        np.mean(total_n_hbonds)))

    if (histogram_path is not None):
        n, bins, patches = plt.hist(total_n_hbonds, 10, facecolor='blue',
                                    alpha=0.5, histtype='bar', ec='black')
        plt.xlabel('Average number of hydrogen bonds per snapshot',
                   fontweight='bold')
        plt.ylabel('Number of simulations', fontweight='bold')
        plt.xlim((0, 10))
        plt.savefig(histogram_path)


if __name__ == "__main__":
    main()
