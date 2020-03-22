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

    parser.add_argument("-m", "--mode", choices=["count",
                                                 "frequent_interactions",
                                                 "relative_frequency"],
                        type=str, metavar="MODE",
                        default="count",
                        help="Selection of computation mode: " +
                        "(1) count - sum of all residues interacting, " +
                        "(2) frequent_interactions - sum of interactions " +
                        "present at least in a 10%% of the structures of" +
                        " the simulation, (3) relative_frequency - mean " +
                        "of interacting residues frequencies for each " +
                        "ligand.")
    parser.add_argument("-l", "--lim",
                        metavar="L", type=float, default='0.1',
                        help="Frequency limit for frequent_interations method")
    parser.add_argument("--epochs_to_ignore", nargs='*',
                        metavar="N", type=int, default=[],
                        help="PELE epochs whose H bonds will be ignored")
    parser.add_argument("--trajectories_to_ignore", nargs='*',
                        metavar="N", type=int, default=[],
                        help="PELE trajectories whose H bonds will be ignored")
    parser.add_argument("--models_to_ignore", nargs='*',
                        metavar="N", type=int, default=[],
                        help="PELE models whose H bonds will be ignored")
    parser.add_argument("-o", "--output",
                        metavar="PATH", type=str, default=None,
                        help="Output path to save the plot")

    args = parser.parse_args()

    return args.hbonds_data_paths, args.mode, args.lim, \
        args.epochs_to_ignore, args.trajectories_to_ignore, \
        args.models_to_ignore, args.output


def create_df(hb_path):
    rows_df = []
    with open(hb_path) as file:
        rows = file.readlines()
        for row in rows[2:]:
            rows_df.append(row.split())
    return rows_df


def get_hbond_atoms_from_df(df, hb_path, epochs_to_ignore,
                            trajectories_to_ignore, models_to_ignore):
    hbond_atoms = []

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
            for residue in residues:
                hbond_atoms.append(residue.split(":"))
        except IndexError:
            pass

    return hbond_atoms


def count(hbond_atoms):
    counter = defaultdict(dict)

    for (chain, residue, atom) in hbond_atoms:
        counter[(chain, residue)][atom] = \
            counter[(chain, residue)].get(atom, 0) + 1

    return counter


def count_norm(hbond_atoms):
    counter = defaultdict(dict)

    if (len(hbond_atoms) == 0):
        return counter

    norm_factor = 1 / len(hbond_atoms)

    for (chain, residue, atom) in hbond_atoms:
        counter[(chain, residue)][atom] = \
            counter[(chain, residue)].get(atom, 0) + 1

    for residue, atom_freq in counter.items():
        for atom, freq in atom_freq.items():
            counter[residue][atom] *= norm_factor

    return counter


def normalize(counter, total):
    for residue, atom_freq in counter.items():
        for atom, freq in atom_freq.items():
            counter[residue][atom] = freq / total

    return counter


def discard_non_frequent(counter, lim=0.1):
    new_counter = defaultdict(dict)

    for (chain, residue), atom_freq in counter.items():
        for atom, freq in atom_freq.items():
            if (freq >= lim):
                new_counter[(chain, residue)][atom] = \
                    new_counter[(chain, residue)].get(atom, 0) + 1

    return new_counter


def combine_results(general_results, mode):
    combined_results = defaultdict(dict)

    if (mode == "count" or mode == "frequent_interactions"):
        for _, hbonds in general_results.items():
            for residue, atom_freq in hbonds.items():
                for atom, freq in atom_freq.items():
                    combined_results[residue][atom] = \
                        combined_results[residue].get(atom, 0) + freq

    elif (mode == "relative_frequency"):
        counter = defaultdict(list)
        atom_set = set()
        for _, hbonds in general_results.items():
            for residue, atom_freq in hbonds.items():
                for atom, freq in atom_freq.items():
                    atom_set.add(residue + (atom, ))

        for _, hbonds in general_results.items():
            for (chain, residue, atom) in atom_set:
                if (chain, residue) in hbonds:
                    if (atom in hbonds[(chain, residue)]):
                        counter[(chain, residue, atom)].append(
                            hbonds[(chain, residue)][atom])
                        continue

                counter[(chain, residue, atom)].append(0)

        for (chain, residue, atom), freqs in counter.items():
            combined_results[(chain, residue)][atom] = np.mean(freqs)

    return combined_results


def generate_barplot(dictionary, mode, lim, output_path):
    fig, ax = plt.subplots(1, figsize=(10, 8))
    fig.tight_layout()
    fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.08)

    # y ticks and labels handlers
    y = 1.4
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

    for residue, atom_freq in sorted(dictionary.items()):
        _ys = []
        jump = False
        for atom, freq in sorted(atom_freq.items()):
            if (freq < max_freq / 100):
                continue
            _ys.append(y)
            sub_ylabels.append(atom)
            sub_xs.append(freq)
            plt.barh(y, freq, align='center', color=cmap(norm(color_index)))
            y += 0.9
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

    plt.ylim((0, y))

    plt.ylabel('COVID-19 Mpro residues', fontweight='bold')
    plt.yticks(ys, ['{}:{}'.format(*i) for i in ylabels])

    if (mode == "count"):
        plt.xlabel('Absolut H bond counts', fontweight='bold')

    elif (mode == "relative_frequency"):
        plt.xlabel('Average of relative H bond frequencies', fontweight='bold')

    elif (mode == "frequent_interactions"):
        plt.xlabel('Absolut H bond counts with frequencies above ' +
                   '{}'.format(lim), fontweight='bold')

    offset = max_freq * 0.025

    for sub_x, sub_y, sub_ylabel in zip(sub_xs, sub_ys, sub_ylabels):
        ax.text(sub_x + offset, sub_y, sub_ylabel.strip(),
                horizontalalignment='left', verticalalignment='center',
                size=7)

    ax.set_facecolor('whitesmoke')

    if (output_path is not None):
        output_path = Path(output_path)

        if (output_path.parent.is_dir()):
            plt.savefig(str(output_path), dpi=300, transparent=True,
                     pad_inches=0.05)
            return

    plt.show()


def main():
    hb_paths, mode, lim, epochs_to_ignore, trajectories_to_ignore, \
        models_to_ignore, output_path = parse_args()

    hb_paths_list = []
    if (type(hb_paths) == list):
        for hb_path in hb_paths:
            hb_paths_list += glob.glob(hb_path)
    else:
        hb_paths_list = glob.glob(hb_paths)
    general_results = {}
    for hb_path in hb_paths_list:
        df = create_df(hb_path)
        hbond_atoms = get_hbond_atoms_from_df(df, hb_path,
                                              epochs_to_ignore,
                                              trajectories_to_ignore,
                                              models_to_ignore)

        if (mode == "count"):
            counter = count(hbond_atoms)

        elif (mode == "relative_frequency"):
            counter = count_norm(hbond_atoms)

        elif mode == "frequent_interactions":
            counter = count_norm(hbond_atoms)
            counter = discard_non_frequent(counter, lim)

        general_results[hb_path] = counter

    combined_results = combine_results(general_results, mode)

    generate_barplot(combined_results, mode, lim, output_path)


if __name__ == "__main__":
    main()
