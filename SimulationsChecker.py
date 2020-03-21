# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
from multiprocessing import Pool

# PELE imports
from Helpers.PELEIterator import SimIt


# Script information
__author__ = "Marti Municoy, Carles Perez"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy, Carles Perez"
__email__ = "marti.municoy@bsc.es, carles.perez@bsc.es"


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("sim_paths", metavar="PATH", type=str,
                        nargs='*',
                        help="Path to PELE simulation folders")
    parser.add_argument("-o", "--PELE_output_path",
                        metavar="PATH", type=str, default='output',
                        help="Relative path to PELE output folder")
    parser.add_argument("-n", "--processors_number",
                        metavar="N", type=int, default=None,
                        help="Number of processors")
    parser.add_argument("-w", "--warning_threshold",
                        metavar="N", type=int, default=500,
                        help="Minimum number of models expected in a normal" +
                        " simulation")

    args = parser.parse_args()

    return args.sim_paths, args.PELE_output_path, args.processors_number, \
        args.warning_threshold


def parallel_models_counter(report):
    with open(str(report), 'r') as f:
        f.readline()
        counter = 0
        for line in f:
            counter += 1

    return counter


def main():
    # Parse args
    PELE_sim_paths, output_path, proc_number, warning_threshold = parse_args()

    all_sim_it = SimIt(PELE_sim_paths)

    for PELE_sim_path in all_sim_it:
        sim_it = SimIt(PELE_sim_path)
        sim_it.build_repo_it(output_path, 'report')

        reports = [repo for repo in sim_it.repo_it]
        with Pool(proc_number) as pool:
            results = pool.map(parallel_models_counter,
                               reports)

        print('Results:')
        for repo, result in zip(reports, results):
            print(' - {:<100}: {:10d} models'.format(str(repo), result))

        print('Warnings:')
        for repo, result in zip(reports, results):
            if (result < warning_threshold):
                print(' - {:<100}: {:10d} models'.format(str(repo), result))


if __name__ == "__main__":
    main()
