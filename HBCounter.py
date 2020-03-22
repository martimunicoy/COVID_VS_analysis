
# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
from multiprocessing import Pool
from functools import partial

# External imports
import mdtraj as md

# PELE imports
from Helpers.PELEIterator import SimIt
from Helpers import hbond_mod as hbm

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
    parser.add_argument("-l", "--ligand_resname",
                        metavar="LIG", type=str, default='LIG',
                        help="Ligand residue name")
    parser.add_argument("-d", "--distance",
                        metavar="D", type=float, default='0.25',
                        help="Hydrogen bonds distance")
    parser.add_argument("-a", "--angle",
                        metavar="A", type=float, default='2.0943951023931953',
                        help="Hydrogen bonds angle")
    parser.add_argument("-p", "--pseudo_hb",
                        metavar="BOOL", type=bool, default=False,
                        help="Look for pseudo hydrogen bonds")
    parser.add_argument("-n", "--processors_number",
                        metavar="N", type=int, default=None,
                        help="Number of processors")
    parser.add_argument("-t", "--topology_path",
                        metavar="PATH", type=str,
                        default='output/topologies/topology_0.pdb',
                        help="Relative path to topology")

    args = parser.parse_args()

    return args.sim_paths, args.ligand_resname, args.distance, args.angle, \
        args.pseudo_hb, args.processors_number, args.topology_path


def find_hbonds_in_trajectory(lig_resname, distance, angle, pseudo,
                              topology_path, chain_ids, traj_path):
    try:
        traj = md.load_xtc(str(traj_path), top=str(topology_path))
    except OSError:
        print('     - Warning: problems loading trajectory '
              '{}, it will be ignored'.format(traj_path))
        return {}

    lig = traj.topology.select('resname {}'.format(lig_resname))
    hbonds_in_traj = find_ligand_hbonds(traj, lig, distance, angle, pseudo,
                                        chain_ids)

    return hbonds_in_traj


def find_ligand_hbonds(traj, lig, distance, angle, pseudo, chain_ids):
    hbonds_dict = {}
    for model_id in range(0, traj.n_frames):
        results = find_hbond_in_snapshot(traj, model_id, lig, distance, angle,
                                         pseudo, chain_ids)
        hbonds_dict[model_id] = results

    return hbonds_dict


def find_hbond_in_snapshot(traj, model_id, lig, distance, angle, pseudo,
                           chain_ids):
    hbonds = hbm.baker_hubbard(traj=traj[model_id], distance=distance,
                               angle=angle, pseudo=pseudo)

    results = []
    for hbond in hbonds:
        if (any(atom in lig for atom in hbond) and not
                all(atom in hbond for atom in lig)):
            for atom in hbond:
                if (atom not in lig):
                    _atom = traj.topology.atom(atom)
                    results.append('{}:{}:{}'.format(
                        chain_ids[_atom.residue.chain.index],
                        _atom.residue,
                        _atom.name))
                    break

    return results


def main():
    # Parse args
    PELE_sim_paths, lig_resname, distance, angle, pseudo_hb, proc_number, \
        topology_relative_path = parse_args()

    all_sim_it = SimIt(PELE_sim_paths)

    print(' - The following PELE simulation paths will be analyzed:')
    for PELE_sim_path in all_sim_it:
        print('   - {}'.format(PELE_sim_path))

    for PELE_sim_path in all_sim_it:
        print(' - Analyzing {}'.format(PELE_sim_path))
        topology_path = PELE_sim_path.joinpath(topology_relative_path)

        if (not topology_path.is_file()):
            print(' - Skipping simulation because topology file with ' +
                  'connectivity was missing')
            continue

        # Retrieve chain ids
        chain_ids = set()
        with open(str(topology_path), 'r') as f:
            for line in f:
                if (len(line) < 80):
                    continue
                line = line.strip()
                chain_ids.add(line[21])
        chain_ids = sorted(list(chain_ids))

        hbonds_dict = {}

        parallel_function = partial(find_hbonds_in_trajectory, lig_resname,
                                    distance, angle, pseudo_hb, topology_path,
                                    chain_ids)

        sim_it = SimIt(PELE_sim_path)
        sim_it.build_traj_it('output', 'trajectory', 'xtc')

        trajectories = [traj for traj in sim_it.traj_it]
        with Pool(proc_number) as pool:
            results = pool.map(parallel_function,
                               trajectories)

        counter = 0
        for r in results:
            counter += len(r.values())

        print('     - {} H bonds were found'.format(counter))

        for r, t in zip(results, trajectories):
            hbonds_dict[int(t.parent.name),
                        int(''.join(filter(str.isdigit, t.name)))] = r

        with open(str(PELE_sim_path.joinpath('hbonds.out')), 'w') as file:
            file.write(str(PELE_sim_path.name) + '\n')
            file.write('Epoch    Trajectory    Model    Hbonds' + '\n')
            for (epoch, traj_num), hbonds in hbonds_dict.items():
                for model, hbs in hbonds.items():
                    if int(epoch) == 0 and int(model) == 0:
                        continue
                    file.write('{:5d}    {:10d}    {:5d}    '.format(
                        epoch, traj_num, model))

                    if (len(hbs) > 0):
                        for hb in hbs[:-1]:
                            file.write('{},'.format(hb))

                        file.write('{}'.format(hbs[-1]))

                    file.write('\n')


if __name__ == "__main__":
    main()
