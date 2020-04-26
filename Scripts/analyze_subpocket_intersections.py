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
from matplotlib import pyplot as plt
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
    parser.add_argument("csv_file", metavar="PATH", type=str,
                        help="Path to subpockets csv file")

    parser.add_argument("traj_paths", metavar="PATH", type=str,
                        nargs='*',
                        help="Path to PELE trajectory files")

    parser.add_argument("-o", "--output_path",
                        metavar="PATH", type=str,
                        default="intersections.csv")

    parser.add_argument("-n", "--processors_number",
                        metavar="N", type=int, default=None,
                        help="Number of processors")

    parser.add_argument("-p", "--PELE_output_path",
                        metavar="PATH", type=str, default='output',
                        help="Relative path to PELE output folder")

    parser.add_argument("-r", "--report_name",
                        metavar="NAME", type=str,
                        default='report',
                        help="PELE report name")

    args = parser.parse_args()

    return args.csv_file, args.traj_paths, args.output_path, \
        args.processors_number, args.PELE_output_path, args.report_name


def analyze_subpocket_intersections(report_csv, report_path, subpockets):
    intersections = defaultdict(list)
    metrics = extract_metrics((report_path, ), (3, ))[0]

    accepted_steps = []
    for m in metrics:
        accepted_steps.append(int(m[0]))

    for a_s in accepted_steps:
        row = report_csv[report_csv['model'] == a_s]
        for sp in subpockets:
            value = float(row['{}_intersection'.format(sp)])
            intersections[sp].append(value)

    return intersections


def main():
    # Parse args
    csv_file, PELE_sim_paths, out_relative_path, proc_number, \
        PELE_output_path, report_name = parse_args()

    csv_file = Path(csv_file)
    out_relative_path = Path(out_relative_path)

    if (not csv_file.is_file()):
        raise ValueError('Wrong path to csv file')

    data = pd.read_csv(csv_file)

    subpockets = []
    for column in data.columns:
        if ('intersection' in column):
            subpockets.append(column.strip('_intersection'))

    if (len(subpockets) == 0):
        raise RuntimeError('No subpocket was found at {}'.format(csv_file))
    else:
        print(' - The following subpocket{} found:'.format(
            [' was', 's were'][len(subpockets) > 1]))
        for subpocket in subpockets:
            print('   - {}'.format(subpocket))

    data['simulation'].fillna('.', inplace=True)
    simulations_in_csv = set(data['simulation'].to_list())

    print(' - {} simulation{} found'.format(
        len(simulations_in_csv),
        [' was', 's were'][len(simulations_in_csv) > 1]))

    all_sim_it = SimIt(PELE_sim_paths)

    simulations_to_analyze = []
    for PELE_sim_path in all_sim_it:
        sim_name = PELE_sim_path.name
        if (sim_name == ''):
            sim_name = '.'
        if (sim_name in simulations_in_csv):
            simulations_to_analyze.append(PELE_sim_path)

    print(' - Simulations that will be analyzed:')
    for sim_path in simulations_to_analyze:
        print('   - {}'.format(sim_path.name))

    for PELE_sim_path in simulations_to_analyze:
        sim_it = SimIt(PELE_sim_path)
        sim_it.build_repo_it(PELE_output_path, report_name)
        print(' - Analyzing {}'.format(PELE_sim_path.name))

        sim_csv = data[data['simulation'] == sim_name]
        reports = [repo for repo in sim_it.repo_it]

        with Pool(proc_number) as pool:
            results = []
            for report in reports:
                epoch = int(report.parent.name)
                trajectory = int(''.join(filter(str.isdigit, report.name)))
                report_csv = sim_csv[(sim_csv['epoch'] == epoch) &
                                     (sim_csv['trajectory'] == trajectory)]
                r = pool.apply(analyze_subpocket_intersections,
                               (report_csv, report, subpockets))
                results.append(r)

        all_intersections = defaultdict(list)
        for r in results:
            for subpocket, intersections in r.items():
                for i in intersections:
                    all_intersections[subpocket].append(i)

        all_intersections = pd.DataFrame.from_dict(all_intersections)
        all_intersections.to_csv(
            str(PELE_sim_path.joinpath(out_relative_path)))


if __name__ == "__main__":
    main()
