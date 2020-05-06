# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
import os
import sys
from typing import Tuple, List, Optional, Union

# External imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpl_patches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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


def parse_args() -> Tuple[List[str], str, str, float, str,
                          Optional[List[float]], Optional[List[int]]]:
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
                        help="Percentile applied to the set of "
                        + "intersections that are obtained for each subpocket",
                        default=95)

    parser.add_argument('-t', '--length_type', default='maximum_steps',
                        nargs='?', type=str,
                        choices=['maximum_steps', 'trajectories_fraction'],
                        help='Type of simulation length to evaluate: '
                        + 'maximum_steps or trajectories_fraction. '
                        + 'Default: maximum_steps')

    parser.add_argument('--trajectories_fraction', metavar="FLOAT", type=float,
                        help="Fraction of trajectories to use in the plot",
                        default=None, nargs='*')

    parser.add_argument('--maximum_steps', metavar="INT", type=int,
                        help="Maximum number of PELE steps to use in the plot",
                        default=None, nargs='*')

    args = parser.parse_args()

    return args.traj_paths, args.subpockets, args.ic50, args.percentile, \
        args.length_type, args.trajectories_fraction, args.maximum_steps


def get_columns(all_sim_it: SimIt, csv_file_name: str) -> list:
    columns = []
    for PELE_sim_path in all_sim_it:
        if (not PELE_sim_path.joinpath(csv_file_name).is_file()):
            print(' - Skipping simulation because subpockets csv file '
                  + 'was missing')
            continue

        data = pd.read_csv(PELE_sim_path.joinpath(csv_file_name))
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        for col in data.columns:
            if ('_intersection' in col and 'nonpolar' not in col):
                if (col not in columns):
                    columns.append(col)

    return columns


def extract_subpocket_results(all_sim_it: SimIt, csv_file_name: str,
                              percentile: float, columns: list,
                              length_type: str,
                              length_iterator: List[Union[float, int]]
                              ) -> pd.DataFrame:
    subpocket_results = pd.DataFrame()

    for PELE_sim_path in all_sim_it:
        print('')
        print(' - Reading data from {}'.format(PELE_sim_path))

        if (not PELE_sim_path.joinpath(csv_file_name).is_file()):
            print(' - Skipping simulation because intersection csv file '
                  + 'is missing')
            continue

        print('   - Retrieving subpocket intersections')
        data = pd.read_csv(PELE_sim_path.joinpath(csv_file_name))

        for length_try in length_iterator:
            # Filtering data frame
            if length_type == 'maximum_steps':
                f_data = data[data['model'] < length_try]
            elif length_type == 'trajectories_fraction':
                all_trajs = list(set(data['trajectory'].to_numpy()))
                selected_trajs = np.random.choice(all_trajs,
                                                  int(length_try
                                                      * len(all_trajs)),
                                                  replace=False)
                f_data = data[subpocket_results['trajectory'].isin(
                    selected_trajs)]
            else:
                raise ValueError(
                    'Invalid length type value: {}'.format(length_type))

            metrics = [PELE_sim_path.name, length_try]
            for col in columns:
                values = f_data[col].values
                metrics.append(np.percentile(values, percentile))

            subpocket_results = pd.concat([subpocket_results,
                                           pd.DataFrame([metrics],
                                                        columns=['path',
                                                                 'length']
                                                        + columns)])

    return subpocket_results


def add_ic50s(subpocket_results: pd.DataFrame, ic50_csv: str
              ) -> pd.DataFrame:
    print(' - Retrieving IC50 values')

    ic50 = pd.read_csv(ic50_csv)
    subpocket_results = subpocket_results.merge(
        ic50, left_on='path', right_on='path')
    subpocket_results['pIC50'] = - np.log10(
        subpocket_results.loc[:, 'IC50'] / 1000000)

    return subpocket_results


def apply_filtering(data: pd.DataFrame, traj_fraction: float,
                    max_steps: Optional[int]) -> pd.DataFrame:
    if max_steps is not None:
        print('   - Filtering by maximum step threshold ({})'.format(
            max_steps))
        new_data = data[data['model'] < max_steps]
        print('     - {} entries were filtered to {}'.format(len(data),
                                                             len(new_data)))
        data = new_data

    if traj_fraction < 1:
        print('   - Filtering by trajectories fraction threshold ({})'.format(
            traj_fraction))
        all_trajs = list(set(data['trajectory'].to_numpy()))
        selected_trajs = np.random.choice(all_trajs,
                                          int(traj_fraction * len(all_trajs)),
                                          replace=False)
        new_data = data[data['trajectory'].isin(selected_trajs)]
        print('     - {} entries were filtered to {}'.format(len(data),
                                                             len(new_data)))
        data = new_data

    return data


def make_plot(all_sim_it: SimIt, csv_file_name: str, ic50_csv: str,
              columns: list, percentile: float, length_type: str,
              traj_fraction: Optional[List[float]],
              max_steps: Optional[List[float]]):
    fig, ax = plt.subplots()
    fig.suptitle('Subpocket-pIC50 correlations vs simulation length')

    ax.set_axisbelow(True)
    ax.grid(True, color='white')
    ax.set_facecolor('lightgray')

    if length_type == 'maximum_steps':
        length_iterator = max_steps
    elif length_type == 'trajectories_fraction':
        length_iterator = traj_fraction
    else:
        raise ValueError('Invalid length type value: {}'.format(length_type))

    subpocket_results = extract_subpocket_results(all_sim_it,
                                                  csv_file_name,
                                                  percentile, columns,
                                                  length_type,
                                                  length_iterator)

    subpocket_results = add_ic50s(subpocket_results, ic50_csv)

    results = []
    for length_try in length_iterator:
        f_subpocket_results = \
            subpocket_results[subpocket_results['length'] == length_try]

        Y = f_subpocket_results.loc[:, columns].to_numpy()
        x = f_subpocket_results['pIC50'].to_numpy()

        current_results = ()
        for y in Y.transpose():
            lin_reg = LinearRegression().fit(y.reshape(-1, 1),
                                             x.reshape(-1, 1))
            x_pred = lin_reg.predict(y.reshape(-1, 1))

            current_results += (r2_score(x, x_pred), )

        results.append(current_results)

    for result, col in zip(zip(*results), columns):
        ax.plot(length_iterator, result, 'x-',
                label=col.strip('_intersection'))

    ax.set_xlabel('Simulation length ({})'.format(
        length_type.replace('_', ' ')))
    ax.set_ylabel('r2 score')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(columns),
              fancybox=True, markerscale=2)

    plt.tight_layout(rect=(0, 0, 0.95, 0.94))
    plt.show()

    return

    """
        subpocket_results = add_ic50s(subpocket_results, ic50_csv)

        Y = subpocket_results.loc[:, columns].to_numpy()
        x = subpocket_results['pIC50'].to_numpy()

        current_results = ()
        for y in Y.transpose():
            lin_reg = LinearRegression().fit(y.reshape(-1, 1),
                                             x.reshape(-1, 1))
            x_pred = lin_reg.predict(y.reshape(-1, 1))

            current_results += (r2_score(x, x_pred), )

        results.append(current_results)

    print(results)

    #ax.plot(r, y)

    # plt.show()
    # for col in columns:
    """
    """
    for i, col in enumerate(columns):
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
    """


def main():
    # Parse args
    PELE_sim_paths, csv_file_name, ic50_csv, percentile, length_type, \
        traj_fraction, max_steps = parse_args()

    all_sim_it = SimIt(PELE_sim_paths)

    print(' - Simulations that will be analyzed:')
    for sim_path in all_sim_it:
        print('   - {}'.format(sim_path.name))

    columns = get_columns(all_sim_it, csv_file_name)

    print(' - Subpockets found:')
    for col in columns:
        print('   - {}'.format(col.strip('_intersection')))

    if (len(columns) == 0):
        raise ValueError('No subpocket intersections were found in the '
                         + 'simulation paths that were supplied')

    make_plot(all_sim_it, csv_file_name, ic50_csv, columns, percentile,
              length_type, traj_fraction, max_steps)


if __name__ == "__main__":
    main()
