# -*- coding: utf-8 -*-


# Standard imports
import os
import argparse as ap
from multiprocessing import Pool
from functools import partial

# External imports
import mdtraj as md

# PELE imports
from Helpers.PELEIterator import SimIt


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("sim_paths", metavar="PATH", type=str,
                        nargs='*',
                        help="Path to PELE simulation folders")
    parser.add_argument("--PELE_output_path",
                        metavar="PATH", type=str, default='output',
                        help="Relative path to PELE output folder")
    parser.add_argument("-n", "--processors_number",
                        metavar="N", type=int, default=None,
                        help="Number of processors")
    parser.add_argument("-o", "--output_path",
                        metavar="PATH", type=str, default="best_metrics")
    parser.add_argument("--ie_col",
                        metavar="N", type=int, default=5,
                        help="Interaction energy column in PELE reports")
    parser.add_argument("-t", "--topology_path",
                        metavar="PATH", type=str,
                        default='output/topologies/topology_0.pdb',
                        help="Relative path to topology")
    parser.add_argument("-l", "--ligand_resname",
                        metavar="LIG", type=str, default='LIG',
                        help="Ligand residue name")

    args = parser.parse_args()

    return args.sim_paths, args.PELE_output_path, args.processors_number, \
        args.output_path, args.ie_col, args.topology_path, args.ligand_resname


def parallel_metrics_getter(ie_col, report):
    tes = []
    ies = []

    with open(str(report), 'r') as f:
        f.readline()
        for line in f:
            line = line.strip()
            fields = line.split()
            tes.append(fields[3])
            ies.append(fields[ie_col - 1])

    return (tes, ies)


def extract_ligand_properties(topology_path, resname):
    n_heavy_atoms = 0
    molecular_weight = 0

    traj = md.load_pdb(str(topology_path))
    lig = traj.topology.select('resname {}'.format(resname))
    for atom in traj.atom_slice(lig).top.atoms:
        if (str(atom.element) != 'hydrogen'):
            n_heavy_atoms += 1
        molecular_weight += atom.element.mass

    return n_heavy_atoms, molecular_weight


def main():
    # Parse args
    PELE_sim_paths, PELE_output_path, proc_number, output_relative_path, \
        ie_col, topology_relative_path, lig_resname = parse_args()

    all_sim_it = SimIt(PELE_sim_paths)

    p_function = partial(parallel_metrics_getter, ie_col)

    for PELE_sim_path in all_sim_it:
        sim_it = SimIt(PELE_sim_path)
        sim_it.build_repo_it(PELE_output_path, 'report')
        print(' - Analyzing {}'.format(PELE_sim_path))

        topology_path = PELE_sim_path.joinpath(topology_relative_path)
        if (not topology_path.is_file()):
            print(' - Skipping simulation because topology file with ' +
                  'connectivity was missing')
            continue

        reports = [repo for repo in sim_it.repo_it]
        with Pool(proc_number) as pool:
            results = pool.map(p_function,
                               reports)

        min_te = 0
        min_ie = 0
        min_te_PDB_id = None
        min_ie_PDB_id = None
        for repo, (tes, ies) in zip(reports, results):
            for i, te in enumerate(tes):
                if (float(te) < min_te):
                    min_te = float(te)
                    min_te_PDB_id = (repo.parent,
                                     int(''.join(filter(str.isdigit,
                                                        repo.name))),
                                     i)
            for i, ie in enumerate(ies):
                if (float(ie) < min_ie):
                    min_ie = float(ie)
                    min_ie_PDB_id = (repo.parent,
                                     int(''.join(filter(str.isdigit,
                                                        repo.name))),
                                     i)

        ligand_heavy_atoms, ligand_mass = extract_ligand_properties(
            topology_path, lig_resname)

        output_path = PELE_sim_path.joinpath(output_relative_path)

        if (not output_path.is_dir()):
            os.mkdir(str(output_path))

        with open(str(output_path.joinpath('results.out')), 'w') as f:
            f.write('lig_heavy_atoms,lig_mass,')
            f.write('best_total_energy,best_interaction_energy\n')
            f.write('{},{:.3f},{},{}\n'.format(ligand_heavy_atoms, ligand_mass,
                                               min_te, min_ie))

        if (min_te_PDB_id is not None):
            t = md.load(str(min_te_PDB_id[0].joinpath(
                'trajectory_{}.xtc'.format(min_te_PDB_id[1]))),
                top=str(topology_path))
            t[min_te_PDB_id[2]].save_pdb(
                str(output_path.joinpath('best_total_energy.pdb')))

        if (min_ie_PDB_id is not None):
            t = md.load(str(min_ie_PDB_id[0].joinpath(
                'trajectory_{}.xtc'.format(min_ie_PDB_id[1]))),
                top=str(topology_path))
            t[min_ie_PDB_id[2]].save_pdb(
                str(output_path.joinpath('best_interaction_energy.pdb')))


if __name__ == "__main__":
    main()
