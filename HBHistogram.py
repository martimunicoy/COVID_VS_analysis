# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
import glob
from collections import defaultdict

# External imports
import matplotlib.pyplot as plt


# Script information
__author__ = "Marti Municoy, Carles Perez"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy, Carles Perez"
__email__ = "marti.municoy@bsc.es, carles.perez@bsc.es"


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("hbonds_data_paths", metavar="PATH", type=str,
                        nargs='*',
                        help="Path to H bonds data files")
    parser.add_argument("-m", "--mode", choices={"count", "frequent_interactions", "relative_frequency"},
                       default="count",
                       help="Selection of computation mode: (1)count - sum of all residues interacting, "
                            "(2)frequent_interactions - sum of interactions present at least in a 10% "
                            "of the structures of the simulation, (3)relative_frequency - mean of interacting "
                            "residues frequencies for each ligand.")

    args = parser.parse_args()

    return args.hbonds_data_paths, args.mode


def create_df(hb_path):
    rows_df = []
    with open(hb_path) as file:
        rows = file.readlines()
        for row in rows[2:]:
            rows_df.append(row.split())
    return rows_df


"""
def get_atoms_from_df(df):
    all_res = []
    for row in df:
        residues = row[3].split(",")[:-1]
        for residue in residues:
            all_res.append(residue)
    return all_res
"""


def get_hbond_atoms_from_df(df):
    hbond_atoms = []
    for row in df:
        residues = row[3].split(",")[:-1]
        for residue in residues:
            hbond_atoms.append(residue.split(":"))
    return hbond_atoms


def count_residues(hbond_atoms):
    counts = {}
    for hbond_atom in hbond_atoms:
        chain, residue, atom = hbond_atom
        counts["{}-{}-{}".format(chain, residue, atom)] = \
            counts.get("{}-{}-{}".format(chain, residue, atom), 0) + 1
    return counts


def count_atoms_from_residue(hbond_atoms):
    residues = []
    res_dict = {}
    for hbond_atom in hbond_atoms:
        chain, residue, atom = hbond_atom
        res_id = "{}-{}".format(chain, residue)
        residues.append(res_id)
    for res in residues:
        atoms_dict = {}
        for hbond_atom in hbond_atoms:
            chain, residue, atom = hbond_atom
            res_id = "{}-{}".format(chain, residue)
            if res == res_id:
                atoms_dict[atom] = atoms_dict.get(atom, 0) + 1
        res_dict[res] = atoms_dict
    return res_dict


def transform_count_to_freq(value, total):
    freq = value / total
    return freq


def count_frequencies(residues_dict, total):
    for name, residue in residues_dict.items():
        for atom, count in residue.items():
            residue[atom] = transform_count_to_freq(count, total)
    return residues_dict


def discard_low_frequent(residues_dict, total, lim=0.1):
    new_dict = {}
    for name, residue in residues_dict.items():
        for atom, count in residue.items():
            freq = transform_count_to_freq(count, total)
            if not freq < lim:
                new_dict.setdefault(name, []).append(atom)
    return new_dict 


def create_barplot(dictionary):
    plt.bar(range(len(dictionary)), list(dictionary.values()), align='center')
    plt.xticks(range(len(dictionary)), list(dictionary.keys()))
    plt.xticks(rotation=45)


def append_simulation_counting(general_dict_results, new_info_dict):
    for name, residue in new_info_dict.items():
        for atom, count in residue.items():
            if not name in general_dict_results.keys():
                res = {}
                res.setdefault(atom, []).append(count)
                general_dict_results[name] = res
            else:
                general_dict_results[name].setdefault(atom, []).append(count)


def join_results(general_dict, mode):
    if mode == "count":
        for resname, residue_dict in general_dict.items():
            for atom, array_of_results in residue_dict.items():        
                residue_dict[atom] = sum(array_of_results)
    if mode == "relative_frequency":
        for resname, residue_dict in general_dict.items():
            for atom, array_of_results in residue_dict.items():
                residue_dict[atom] = sum(array_of_results) / len(array_of_results)


def main():
    hb_paths, mode = parse_args()

    hb_paths_list = []
    if (type(hb_paths) == list):
        for hb_path in hb_paths:
            hb_paths_list += glob.glob(hb_path)
    else:
        hb_paths_list = glob.glob(hb_paths)
    general_results = {}
    for hb_path in hb_paths_list:
        df = create_df(hb_path)
        hbond_atoms = get_hbond_atoms_from_df(df)
        total = len(hbond_atoms)
        counting = count_atoms_from_residue(hbond_atoms)
        if mode == "relative_frequency":
            counting = count_frequencies(counting, total)
        append_simulation_counting(general_results, counting)
        if mode == "frequent_interactions":
            counting = discard_low_frequent(counting, total, lim=0.1)
    join_results(general_results, mode)


if __name__ == "__main__":
    main()
