# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
import glob

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

    args = parser.parse_args()

    return args.hbonds_data_paths


def create_df(hb_paths):
    rows_df = []

    for hb_path in hb_paths:
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


def create_barplot(dictionary):
    plt.bar(range(len(dictionary)), list(dictionary.values()), align='center')
    plt.xticks(range(len(dictionary)), list(dictionary.keys()))
    plt.xticks(rotation=90)


def main():
    hb_paths = parse_args()

    hb_paths_list = []
    if (type(hb_paths) == list):
        for hb_path in hb_paths:
            hb_paths_list += glob.glob(hb_path)
    else:
        hb_paths_list = glob.glob(hb_paths)

    df = create_df(hb_paths_list)
    hbond_atoms = get_hbond_atoms_from_df(df)
    counting = count_residues(hbond_atoms)
    create_barplot(counting)
    plt.savefig("test.png")


if __name__ == "__main__":
    main()
