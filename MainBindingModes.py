# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
from multiprocessing import Pool
from functools import partial
from collections import defaultdict
from operator import itemgetter

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

# Variable
OUT_PATTERN = "output/{}/trajectory_{}.xtc"

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

    parser.add_argument("--hbonds_path",
                        metavar="PATH", type=str,
                        default='hbonds.out',
                        help="Path to H bonds file")

    parser.add_argument("-g", "--golden_hbonds", nargs='*',
                        metavar="C:R[:A]", type=str, default=[],
                        help="Chain (C), residue (R) [and atom (A)] of golden "
                        "H bonds that will be the main filtering condition")

    args = parser.parse_args()

    return args.traj_paths, args.ligand_resname, args.bandwidth, \
        args.processors_number, args.ie_col, args.rmsd_col, \
        args.topology_path, args.report_name, args.hbonds_path, \
        args.golden_hbonds


def extract_epochs_and_traj_nums(trajectories):
    epochs = []
    traj_nums = []

    for traj in trajectories:
        epochs.append(int(traj.parent.name))
        traj_nums.append(int(''.join(filter(str.isdigit, traj.name))))

    return epochs, traj_nums


def extract_ligand_coords(epochs, traj_nums, trajectories, lig_resname,
                          topology_path, proc_number):
    parallel_function = partial(p_extract_ligand_coords,
                                lig_resname,
                                topology_path)

    with Pool(proc_number) as pool:
        results = pool.map(parallel_function,
                           trajectories)

    PELE_ids = []
    for e, t, r in zip(epochs, traj_nums, results):
        for i in range(0, len(r)):
            PELE_ids.append((e, t, i))

    return np.concatenate(results), PELE_ids


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

    for cluster, probability in p_dict.items():
        p_dict[cluster] /= total

    return p_dict


def filter_structures_by_cluster(cluster_results, coordinates, ids):
    struct_dict = defaultdict(list)
    for cluster, coords, id in zip(cluster_results, coordinates, ids):
        coords = np.reshape(coords, (-1, 3))
        struct_dict[cluster].append((coords, id))

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


def extract_hbonds(hbonds_path):
    hbonds = defaultdict(list)

    with open(str(hbonds_path), 'r') as file:
        # Skip two header lines
        file.readline()
        file.readline()

        # Extra hbonds and construct dict
        for line in file:
            line = line.strip()
            fields = line.split()
            epoch, trajectory, model = map(int, fields[:3])
            _hbonds = []
            try:
                for hb in fields[3].split(','):
                    _hbonds.append(hb)
            except IndexError:
                pass
            hbonds[(epoch, trajectory, model)] = tuple(_hbonds)

    return hbonds


def ignore_bonding_atom(hbonds):
    for PELE_id, _hbonds in hbonds.items():
        new_hbonds = []
        for hb in _hbonds:
            new_hbonds.append(':'.join([i for i in hb.split(':')[:2]]))
        hbonds[PELE_id] = tuple(new_hbonds)

    return hbonds


def calculate_hbonds_freq(results, PELE_ids, hbonds):
    # Get hbonds by cluster
    hbonds_by_cluster = defaultdict(list)
    for cluster, PELE_id in zip(results, PELE_ids):
        for hb in hbonds[PELE_id]:
            hbonds_by_cluster[cluster].append(hb)

    # Calculate frequencies
    hbond_freqs = defaultdict(dict)
    for cluster, _hbonds in hbonds_by_cluster.items():
        counter = {}
        for hb in set(_hbonds):
            counter[hb] = _hbonds.count(hb)

        norm_factor = 1 / len(_hbonds)

        for hb, num in counter.items():
            hbond_freqs[cluster][hb] = num * norm_factor

    return hbond_freqs


def get_best_binding_mode(hbond_freqs, golden_hbonds, p_dict):
    score_by_cluster = {}
    for cluster, _hbond_freqs in hbond_freqs.items():
        # Skip low-populated clusters
        if (p_dict[cluster] < 0.01):
            continue
        score_by_cluster[cluster] = 0
        for hbond, freq in _hbond_freqs.items():
            if (hbond in golden_hbonds):
                score_by_cluster[cluster] += freq

    print(score_by_cluster)

    return sorted(score_by_cluster.items(), key=itemgetter(1), reverse=True)[0]


def write_pdb_from_coords(trajectory, new_coords, outfile="test.pdb", ligname="LIG"):
    lig_top = trajectory.topology.select("resname {}".format(ligname))
    lig = trajectory[0].xyz
    ligand_snap.save_pdb(outfile)


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
        ie_col, rmsd_col, topology_relative_path, report_name, \
        hbonds_path, golden_hbonds = parse_args()

    all_sim_it = SimIt(PELE_sim_paths)

    for PELE_sim_path in all_sim_it:
        print(' - Extracting ligand coords from {}'.format(PELE_sim_path))
        hbonds_path = PELE_sim_path.joinpath(hbonds_path)
        topology_path = PELE_sim_path.joinpath(topology_relative_path)

        if (not topology_path.is_file()):
            print(' - Skipping simulation because topology file with ' +
                  'connectivity was missing')
            continue

        if (not hbonds_path.is_file()):
            print(' - Skipping simulation because hbonds file was ' +
                  'missing')
            continue

        sim_it = SimIt(PELE_sim_path)
        sim_it.build_traj_it('output', 'trajectory', 'xtc')

        trajectories = [traj for traj in sim_it.traj_it]
        epochs, traj_nums = extract_epochs_and_traj_nums(trajectories)

        lig_coords, PELE_ids = extract_ligand_coords(epochs, traj_nums,
                                                     trajectories, lig_resname,
                                                     topology_path,
                                                     proc_number)

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
        #print(p_dict)
        struct_dict = filter_structures_by_cluster(results, lig_coords, 
                                                   PELE_ids)
        rmsds_dict = calculate_rmsds(results, rmsds)
        #print(rmsds_dict)
        mean_rmsds_dict, std_rmsds_dict = \
            calculate_mean_and_std_rmsds(rmsds_dict)
        # best_clusters = select_best_clusters()

        hbonds = extract_hbonds(hbonds_path)

        if (len(golden_hbonds) > 0 and (golden_hbonds[0]) == 2):
            hbonds = ignore_bonding_atom(hbonds)

        hbond_freqs = calculate_hbonds_freq(results, PELE_ids, hbonds)

        try:
            best_binding_mode = get_best_binding_mode(hbond_freqs,
                                                      golden_hbonds,
                                                      p_dict)
        except IndexError:
            best_binding_mode = None
        epoch, trajnum, snap = struct_dict[best_binding_mode[0]][0][1] 
        # Obtain the ID
        trajectory = OUT_PATTERN.format(epoch, trajnum, snap)
        # Get the name of the trajectory path
        try:
            traj = md.load(str(trajectory), top=topology_relative_path)
        except FileNotFoundError:
            raise("The trajectory file {} does not exist".format(trajectory))
        # Load the trajectory in mdtraj and finally export a PDB file
        traj[snap].save_pdb("best_structure.pdb") 

        # Extract representative structure of the best binding mode and check
        # H bonds! B:GLY302:O




        #print(p_dict, struct_dict, rmsds_dict, mean_rmsds_dict, std_rmsds_dict)







if __name__ == "__main__":
    main()
