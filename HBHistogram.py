# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
import glob
from collections import defaultdict

# External imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

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
    parser.add_argument("-m", "--mode", choices={"count",
                                                 "frequent_interactions",
                                                 "relative_frequency"},
                        default="count",
                        help="Selection of computation mode: " +
                        "(1)count - sum of all residues interacting, " +
                        "(2)frequent_interactions - sum of interactions " +
                        "present at least in a 10\% of the structures of" +
                        " the simulation, (3)relative_frequency - mean " +
                        "of interacting residues frequencies for each " +
                        "ligand.")

    args = parser.parse_args()

    return args.hbonds_data_paths, args.mode


def create_df(hb_path):
    rows_df = []
    with open(hb_path) as file:
        rows = file.readlines()
        for row in rows[2:]:
            rows_df.append(row.split())
    return rows_df


def get_hbond_atoms_from_df(df):
    hbond_atoms = []
    for row in df:
        try:
            residues = row[3].split(",")[:-1]
            for residue in residues:
                hbond_atoms.append(residue.split(":"))
        except IndexError:
            pass
    return hbond_atoms


def count(hbond_atoms):
    counter = defaultdict(dict)
    for hbond_atom in hbond_atoms:
        chain, residue, atom = hbond_atom
        counter[(chain, residue)][atom] = \
            counter[(chain, residue)].get(atom, 0) + 1
    return counter


def transform_count_to_freq(value, total):
    freq = value / total
    return freq


def normalize(counter, total):
    for residue, atom_freq in counter.items():
        for atom, freq in atom_freq.items():
            counter[residue][atom] = freq / total

    return counter


def discard_low_frequent(residues_dict, total, dict_to_update=None, lim=0.1):
    for name, residue in residues_dict.items():
        for atom, count in residue.items():
            freq = transform_count_to_freq(count, total)

        if not freq < lim:
            if not name in dict_to_update.keys():
                dictionary = {}
                dictionary[atom] = dictionary.get(atom, 0) + 1
                dict_to_update[name] = dictionary
            else:
                dict_to_update[name][atom] = dict_to_update[name].get(atom, 0) + 1


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


def combine_results(general_results, mode):
    combined_results = defaultdict(dict)

    if (mode == "count"):
        for _, hbonds in general_results.items():
            for residue, atom_freq in hbonds.items():
                for atom, freq in atom_freq.items():
                    combined_results[residue][atom] = \
                        combined_results[residue].get(atom, 0) + freq

    elif (mode == "relative_frequency"):
        counter = defaultdict(list)
        for _, hbonds in general_results.items():
            for residue, atom_freq in hbonds.items():
                for atom, freq in atom_freq.items():
                    counter[residue + (atom, )].append(freq)

        for (chain, residue, atom), freqs in counter.items():
            combined_results[residue][atom] = np.mean(freqs)

    return combined_results


def counter_barplot(dictionary):
    fig, ax = plt.subplots(1)

    # y ticks and labels handlers
    y = 0.5
    ys = []
    sub_ys = []
    sub_xs = []
    ylabels = []
    sub_ylabels = []

    # colormap handlers
    norm = matplotlib.colors.Normalize(0, 10)
    cmap = cm.get_cmap('tab10')
    color_index = 0

    max_freq = 0
    for residue, atom_freq in dictionary.items():
        for atom, freq in atom_freq.items():
            if (freq > max_freq):
                max_freq = freq

    for residue, atom_freq in dictionary.items():
        _ys = []
        jump = False
        for atom, freq in atom_freq.items():
            if (freq < max_freq / 100):
                continue
            _ys.append(y)
            sub_ylabels.append(atom)
            sub_xs.append(freq)
            plt.barh(y, freq, align='center', color=cmap(norm(color_index)))
            y += 1
            jump = True

        if (jump):
            ys.append(np.mean(_ys))
            ylabels.append(residue)
            sub_ys += _ys
            if (color_index < 9):
                color_index += 1
            else:
                color_index = 0
            y += 0.5

    plt.ylabel('Residues', fontweight='bold')
    plt.yticks(ys, ['{}:{}'.format(*i) for i in ylabels])

    offset = max_freq * 0.025

    for sub_x, sub_y, sub_ylabel in zip(sub_xs, sub_ys, sub_ylabels):
        ax.text(sub_x + offset, sub_y, sub_ylabel.strip(),
                horizontalalignment='right', verticalalignment='center',
                size=7)

    ax.set_facecolor('whitesmoke')

    plt.show()


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
        counter = count(hbond_atoms)

        if (mode == "relative_frequency"):
            counter = normalize(counter, total)
        """
        if mode == "frequent_interactions":
            discard_low_frequent(counter, total, general_results, lim=0.1)
        """
        general_results[hb_path] = counter
    """
    if mode != "frequent_interactions":
        join_results(general_results, mode)
    """
    combined_results = combine_results(general_results, mode)

    counter_barplot(combined_results)


if __name__ == "__main__":
    main()
