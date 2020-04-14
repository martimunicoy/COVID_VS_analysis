# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
import glob
from collections import defaultdict
from pathlib import Path

# External imports
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# PELE imports
from Helpers.PELEIterator import SimIt
from Helpers.ReportUtils import extract_PELE_ids
from Helpers.ReportUtils import extract_metrics
from Helpers.ReportUtils import get_metric_by_PELE_id


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
                                                 "relative_frequency",
                                                 "mean_energies"],
                        type=str, metavar="MODE",
                        default="count",
                        help="Selection of computation mode: " +
                        "(1) count - sum of all residues interacting, " +
                        "(2) frequent_interactions - sum of interactions " +
                        "present at least in a 10%% of the structures of" +
                        " the simulation, (3) relative_frequency - mean " +
                        "of interacting residues frequencies for each " +
                        "ligand, (4) mean_energies - mean interaction " +
                        "energies per H bond are calculated")
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
    parser.add_argument("-n", "--processors_number",
                        metavar="N", type=int, default=None,
                        help="Number of processors")
    parser.add_argument("--PELE_output_path",
                        metavar="PATH", type=str, default='output',
                        help="Relative path to PELE output folder")
    parser.add_argument("--PELE_report_name",
                        metavar="PATH", type=str, default='report',
                        help="Name of PELE's reports")

    args = parser.parse_args()

    return args.hbonds_data_paths, args.mode, args.lim, \
        args.epochs_to_ignore, args.trajectories_to_ignore, \
        args.models_to_ignore, args.output, args.processors_number, \
        args.PELE_output_path, args.PELE_report_name


def create_df(hb_path):
    rows_df = []
    with open(hb_path) as file:
        rows = file.readlines()
        for row in rows[2:]:
            rows_df.append(row.split())
    return rows_df


def get_hbond_atoms_from_df(df, hb_path, epochs_to_ignore,
                            trajectories_to_ignore, models_to_ignore):
    hbond_atoms = defaultdict(list)

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
                hbond_atoms[(epoch, trajectory, model)].append(
                    residue.split(":"))
        except IndexError:
            pass

    return hbond_atoms


def count(hbond_atoms):
    counter = defaultdict(dict)

    for _, hbonds in hbond_atoms.items():
        for (chain, residue, atom) in hbonds:
            counter[(chain, residue)][atom] = \
                counter[(chain, residue)].get(atom, 0) + 1

    return counter


def count_norm(hbond_atoms):
    counter = defaultdict(dict)

    if (len(hbond_atoms) == 0):
        return counter

    number_of_snapshots = len(hbond_atoms)
    norm_factor = 1 / number_of_snapshots

    for _, hbonds in hbond_atoms.items():
        for (chain, residue, atom) in hbonds:
            counter[(chain, residue)][atom] = \
                counter[(chain, residue)].get(atom, 0) + 1

    for residue, atom_freq in counter.items():
        for atom, freq in atom_freq.items():
            counter[residue][atom] *= norm_factor

    return counter


def discard_non_frequent(counter, lim=0.1):
    new_counter = defaultdict(dict)

    for (chain, residue), atom_freq in counter.items():
        for atom, freq in atom_freq.items():
            if (freq >= lim):
                new_counter[(chain, residue)][atom] = \
                    new_counter[(chain, residue)].get(atom, 0) + 1

    return new_counter


def count_energy(hbond_atoms, ie_by_PELE_id):
    counter = defaultdict(lambda: defaultdict(list))
    for PELE_id, hbs in hbond_atoms.items():
        # Preventing repeated H bonds in the same snapshot
        for (chain, residue, atom) in set(map(tuple, hbs)):
            counter[(chain, residue)][atom].append(ie_by_PELE_id[PELE_id])

    # Calculate mean and sum of means
    sum_of_means = float(0.0)
    for (chain, residue), atom_ies in counter.items():
        for atom, ies in atom_ies.items():
            ies_mean = np.mean(ies)
            counter[(chain, residue)][atom] = ies_mean
            sum_of_means += ies_mean

    if (sum_of_means == 0):
        return defaultdict(dict)

    norm_factor = 1 #/ sum_of_means

    # Calculate relative energy
    for (chain, residue), atom_ies in counter.items():
        for atom, ies in atom_ies.items():
            counter[(chain, residue)][atom] *= norm_factor

    return counter


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

    elif (mode == "mean_energies"):
        ie_combiner = defaultdict(lambda: defaultdict(list))
        atom_set = set()
        for _, hbonds in general_results.items():
            for residue, atom_ies in hbonds.items():
                for atom, ie in atom_ies.items():
                    ie_combiner[residue][atom].append(ie)

        for residue, atom_ies in ie_combiner.items():
            for atom, ies in atom_ies.items():
                combined_results[residue][atom] = np.mean(ies)

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
    norm = mpl.colors.Normalize(0, 10)
    cmap = cm.get_cmap('tab10')
    color_index = 0

    max_freq = None
    min_freq = None
    for residue, atom_freq in dictionary.items():
        for atom, freq in atom_freq.items():
            if (max_freq is None or freq > max_freq):
                max_freq = freq
            if (min_freq is None or freq < min_freq):
                min_freq = freq

    for residue, atom_freq in sorted(dictionary.items()):
        _ys = []
        jump = False
        for atom, freq in sorted(atom_freq.items()):
            if (mode != 'mean_energies'):
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

    elif (mode == "mean_energies"):
        ax.set_xlim(max_freq + (max_freq - min_freq) * 0.05,
                    min_freq - (max_freq - min_freq) * 0.05)
        plt.xlabel('Average of mean total energies for each H bond',
                   fontweight='bold')

    if (mode == 'mean_energies'):
        offset = 0
    else:
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
        models_to_ignore, relative_output_path, proc_number, \
        PELE_output_path, PELE_report_name = parse_args()

    hb_paths_list = []
    if (type(hb_paths) == list):
        for hb_path in hb_paths:
            hb_paths_list += glob.glob(hb_path)
    else:
        hb_paths_list = glob.glob(hb_paths)
    general_results = {}
    for hb_path in hb_paths_list:
        df = create_df(hb_path)
        # Calculate hbond_atoms, which is a dict with PELE_ids as key and
        # corresponding lists of H bonds as values
        hbond_atoms = get_hbond_atoms_from_df(df, hb_path,
                                              epochs_to_ignore,
                                              trajectories_to_ignore,
                                              models_to_ignore)

        output_path = Path(str(Path(hb_path).parent) + relative_output_path)

        if (mode == "count"):
            counter = count(hbond_atoms)

        elif (mode == "relative_frequency"):
            counter = count_norm(hbond_atoms)

        elif (mode == "frequent_interactions"):
            counter = count_norm(hbond_atoms)
            counter = discard_non_frequent(counter, lim)

        elif (mode == "mean_energies"):
            sim_it = SimIt(Path(hb_path).parent)
            sim_it.build_repo_it(PELE_output_path, 'report')
            reports = [repo for repo in sim_it.repo_it]

            PELE_ids = extract_PELE_ids(reports)
            metrics = extract_metrics(reports, (4, ), proc_number)

            ies = []
            for ies_chunk in metrics:
                ies.append(list(map(float, np.concatenate(ies_chunk))))

            ie_by_PELE_id = get_metric_by_PELE_id(PELE_ids, ies)

            counter = count_energy(hbond_atoms, ie_by_PELE_id)

        general_results[hb_path] = counter

    combined_results = combine_results(general_results, mode)

    generate_barplot(combined_results, mode, lim, output_path)


if __name__ == "__main__":
    main()
