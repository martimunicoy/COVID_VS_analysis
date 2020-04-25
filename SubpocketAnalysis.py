# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
from functools import partial
from multiprocessing import Pool

# PELE imports
from Helpers.PELEIterator import SimIt
from Helpers.SubpocketAnalysis import ChainConverterForMDTraj, Subpocket
from Helpers.SubpocketAnalysis import build_residues

# External imports
import mdtraj as md
import pandas as pd
import numpy as np


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("traj_paths", metavar="PATH", type=str,
                        nargs='*',
                        help="Path to PELE trajectory files")

    parser.add_argument("-l", "--ligand_resname",
                        metavar="LIG", type=str, default='LIG',
                        help="Ligand residue name")

    parser.add_argument("-t", "--topology_path",
                        metavar="PATH", type=str,
                        default='output/topologies/topology_0.pdb',
                        help="Relative path to topology")

    parser.add_argument("-r", "--report_name",
                        metavar="NAME", type=str,
                        default='report',
                        help="PELE report name")

    parser.add_argument("-s", "--subpocket", nargs='*', action='append',
                        metavar="C:R", type=str, default=[],
                        help="Chain (C), residue (R) of the subset of " +
                        "the residues that define the subpocket")

    parser.add_argument("--subpocket_names", nargs='*',
                        metavar="NAME", type=str, default=None,
                        help="Name of each subpocket")

    parser.add_argument("-p", "--probe_atom_name",
                        metavar="NAME", type=str, default='CA',
                        help="Name of probe atom that will be used to " +
                        "define the subpocket")

    parser.add_argument("-n", "--processors_number",
                        metavar="N", type=int, default=None,
                        help="Number of processors")

    parser.add_argument("-o", "--output_path",
                        metavar="PATH", type=str, default="subpockets.csv")

    parser.add_argument("--PELE_output_path",
                        metavar="PATH", type=str, default='output',
                        help="Relative path to PELE output folder")

    args = parser.parse_args()

    return args.traj_paths, args.ligand_resname, args.topology_path, \
        args.report_name, args.subpocket, args.subpocket_names, \
        args.probe_atom_name, args.processors_number, args.output_path, \
        args.PELE_output_path


def subpocket_analysis(sim_path, subpockets, topology_path, trajectory):
    results = []
    for i, snapshot in enumerate(md.load(str(trajectory),
                                         top=str(topology_path))):
        entry = []
        entry.append(sim_path.name)
        entry.append(int(trajectory.parent.name))
        entry.append(int(''.join(filter(str.isdigit,
                         trajectory.name))))
        entry.append(i)
        for subpocket in subpockets:
            entry.append(subpocket.get_centroid(snapshot))

        results.append(entry)

    return results


def main():
    # Parse args
    PELE_sim_paths, lig_resname, topology_relative_path, report_name, \
        subpockets_residues, subpocket_names, probe_atom_name, proc_number, \
        output_path, PELE_output_path = parse_args()

    all_sim_it = SimIt(PELE_sim_paths)

    data = pd.DataFrame()

    for PELE_sim_path in all_sim_it:
        topology_path = PELE_sim_path.joinpath(topology_relative_path)

        if (not topology_path.is_file()):
            print(' - Skipping simulation because topology file with ' +
                  'connectivity was missing')
            continue

        chain_converter = ChainConverterForMDTraj(str(topology_path))

        subpockets = []
        for subpocket_residues in subpockets_residues:
            residues = build_residues([(i.split(':')[0], int(i.split(':')[1]))
                                      for i in subpocket_residues],
                                      chain_converter)

            subpockets.append(Subpocket(residues))

        sim_it = SimIt(PELE_sim_path)
        sim_it.build_traj_it(PELE_output_path, 'trajectory', 'xtc')

        trajectories = [traj for traj in sim_it.traj_it]

        if (subpocket_names is not None and
                len(subpocket_names) != len(subpockets)):
            print(' - Warning: length of subpocket_names does not match ' +
                  'with length of subpockets, custom names are ignored.')
            subpocket_names = None

        if (subpocket_names is None):
            subpocket_names = []
            for i, subpocket in enumerate(subpockets):
                subpocket_names.append('S{}'.format(i + 1))

        parallel_function = partial(subpocket_analysis, PELE_sim_path,
                                    subpockets, topology_path)

        with Pool(proc_number) as pool:
            results = pool.map(parallel_function, trajectories)

        data = pd.concat(
            [data, ] +
            [pd.DataFrame([r],
             columns=['simulation', 'epoch', 'trajectory', 'model'] +
                ["{}_centroid".format(i) for i in subpocket_names])
             for r in np.concatenate(results)])

    data.to_csv(output_path)


if __name__ == "__main__":
    main()
