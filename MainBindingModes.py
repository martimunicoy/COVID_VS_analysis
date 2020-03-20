# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
from multiprocessing import Pool
from functools import partial
from collections import defaultdict

# External imports
import mdtraj as md
import numpy as np
from sklearn.cluster import MeanShift

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
    parser.add_argument("traj_paths", metavar="PATH", type=str,
                        nargs='*',
                        help="Path to PELE trajectory files")

    parser.add_argument("-l", "--ligand_resname",
                        metavar="LIG", type=str, default='LIG',
                        help="Ligand residue name")

    parser.add_argument("-b", "--bandwidth",
                        metavar="B", type=float, default='5',
                        help="Clustering bandwidth")

    parser.add_argument("-n", "--processors_number",
                        metavar="N", type=int, default=None,
                        help="Number of processors")

    parser.add_argument("--ie_col",
                        metavar="N", type=int, default=5,
                        help="Interaction energy column in PELE reports")

    parser.add_argument("--rmsd_col",
                        metavar="N", type=int, default=7,
                        help="RMSD column in PELE reports")

    parser.add_argument("-t", "--topology_path",
                        metavar="PATH", type=str,
                        default='output/topologies/topology_0.pdb',
                        help="Relative path to topology")

    parser.add_argument("-r", "--report_name",
                        metavar="NAME", type=str,
                        default='report',
                        help="PELE report name")

    args = parser.parse_args()

    return args.traj_paths, args.ligand_resname, args.bandwidth, \
        args.processors_number, args.ie_col, args.rmsd_col, \
        args.topology_path, args.report_name


def extract_ligand_coords(trajectories, lig_resname, topology_path,
                          proc_number):
    parallel_function = partial(p_extract_ligand_coords,
                                lig_resname,
                                topology_path)

    with Pool(proc_number) as pool:
        results = pool.map(parallel_function,
                           trajectories)

    return np.concatenate(results)


def p_extract_ligand_coords(lig_resname, topology_path, traj_path):
    try:
        traj = md.load_xtc(str(traj_path), top=str(topology_path))
    except OSError:
        print('     - Warning: problems loading trajectory '
              '{}, it will be ignored'.format(traj_path))
        return {}

    lig = traj.topology.select('resname {}'.format(lig_resname))

    lig_coords = traj.atom_slice(lig).xyz

    reshaped_lig_coords = []
    for chunk in lig_coords:
        reshaped_lig_coords.append(np.concatenate(chunk))

    return reshaped_lig_coords


def clusterize(lig_coords, bandwidth, proc_number):
    print(' - Clustering')
    gm = MeanShift(bandwidth=bandwidth,
                   n_jobs=proc_number,
                   cluster_all=True)

    return gm.fit_predict(lig_coords)


def extract_ligand_metrics(trajectories, report_name, cols, proc_number):
    parallel_function = partial(p_extract_ligand_metrics,
                                report_name, cols)

    with Pool(proc_number) as pool:
        results = pool.map(parallel_function,
                           trajectories)

    return results


def p_extract_ligand_metrics(report_name, cols, traj):
    # Recover corresponding report
    num = int(''.join(filter(str.isdigit, traj.name)))
    path = traj.parent

    report_path = path.joinpath(report_name + '_{}'.format(num))

    results = []

    if (report_path.is_file()):
        try:
            with open(str(report_path), 'r') as f:
                f.readline()
                for line in f:
                    line.strip()
                    fields = line.split()
                    metrics = []
                    for col in cols:
                        metrics.append(fields[col - 1])
                    results.append(metrics)

        except IndexError:
            print(' - p_extract_ligand_metric Warning: wrong index ' +
                  'supplied for trajectory: \'{}\''.format(traj))
    else:
        print(' - p_extract_ligand_metric Warning: wrong path to report ' +
              'for trajectory: \'{}\''.format(traj))

    return results


def calculate_probabilities(cluster_results):
    p_dict = defaultdict(int)
    total = 0
    for cluster in cluster_results:
        p_dict[cluster] += 1
        total += 1

    return p_dict


def filter_structures_by_cluster(cluster_results, coordinates):
    struct_dict = defaultdict(list)
    for cluster, coords in zip(cluster_results, coordinates):
        coords = np.reshape(coords, (-1, 3))
        struct_dict[cluster].append(coords)

    return struct_dict


def calculate_rmsds(results, rmsds):
    rmsds_dict = defaultdict(list)
    for cluster, rmsd in zip(results, rmsds):
        rmsds_dict[cluster].append(rmsd)

    return rmsds_dict


def calculate_mean_and_std_rmsds(rmsds_dict):
    mean_rmsds_dict = {}
    std_rmsds_dict = {}
    for cluster, rmsds in rmsds_dict.items():
        mean_rmsds_dict[cluster] = np.mean(rmsds)
        std_rmsds_dict[cluster] = np.std(rmsds)

    return mean_rmsds_dict, std_rmsds_dict


"""
def select_best_clusters(results, ):
    best_clusters = [i for i, j in sorted(mean_ie_dict.items(), key=lambda item: item[1])]
    best_results = []
    new_ids = {}
    for bc in best_clusters:
        if (p_dict[bc] / total < lowest_density):
            continue
        if (len(new_ids) > number_of_best_clusters - 1):
            break
        new_ids[bc] = len(new_ids)

    for r in results:
        if (r in new_ids.keys()):
            best_results.append(new_ids[r])
        else:
            best_results.append(-1)
"""


def main():
    # Parse args
    PELE_sim_paths, lig_resname, bandwidth, proc_number, \
        ie_col, rmsd_col, topology_relative_path, report_name = parse_args()

    all_sim_it = SimIt(PELE_sim_paths)

    for PELE_sim_path in all_sim_it:
        print(' - Extracting ligand coords from {}'.format(PELE_sim_path))
        topology_path = PELE_sim_path.joinpath(topology_relative_path)

        if (not topology_path.is_file()):
            print(' - Skipping simulation because topology file with ' +
                  'connectivity was missing')
            continue

        sim_it = SimIt(PELE_sim_path)
        sim_it.build_traj_it('output', 'trajectory', 'xtc')

        trajectories = [traj for traj in sim_it.traj_it]

        lig_coords = extract_ligand_coords(trajectories, lig_resname,
                                           topology_path, proc_number)

        results = clusterize(lig_coords, bandwidth, proc_number)

        metrics = extract_ligand_metrics(trajectories, report_name,
                                         (ie_col, rmsd_col),
                                         proc_number)

        ies = []
        rmsds = []
        for chunk in metrics:
            for ie, rmsd in chunk:
                ies.append(float(ie))
                rmsds.append(float(rmsd))

        p_dict = calculate_probabilities(results)
        print(p_dict)
        struct_dict = filter_structures_by_cluster(results, lig_coords)
        rmsds_dict = calculate_rmsds(results, rmsds)
        print(rmsds_dict)
        mean_rmsds_dict, std_rmsds_dict = \
            calculate_mean_and_std_rmsds(rmsds_dict)
        # best_clusters = select_best_clusters()

        print(p_dict, struct_dict, rmsds_dict, mean_rmsds_dict, std_rmsds_dict)







if __name__ == "__main__":
    main()
