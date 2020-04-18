# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
from collections import defaultdict

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
    parser.add_argument("traj_paths", metavar="PATH", type=str,
                        nargs='*',
                        help="Path to PELE trajectory files")

    parser.add_argument("--hbonds_path",
                        metavar="PATH", type=str,
                        default='hbonds.out',
                        help="Path to H bonds file")

    parser.add_argument("--hbonds", nargs='*',
                        metavar="C:R[:A1, A2]", type=str, default=[],
                        help="List of H bonds whose persistance will be " +
                        "calculated")

    parser.add_argument("-o", "--output_path",
                        metavar="PATH", type=str, default="persistance.out")

    parser.add_argument("-l", "--ligand_resname",
                        metavar="LIG", type=str, default='LIG',
                        help="Ligand residue name")

    args = parser.parse_args()

    return args.traj_paths, args.hbonds_path, args.hbonds, args.output_path, \
        args.ligand_resname


def prepare_hbond_dict(hbonds):
    hbond_dict = {}
    for hb in hbonds:
        hb_data = hb.split(':')
        if (len(hb_data) == 2):
            hbond_dict[tuple(hb_data)] = ['all']
        elif (len(hb_data) == 3):
            hbond_dict[tuple(hb_data[0:2])] = hb_data[2].split(',')
        else:
            print('Error: H bond \'{}\' have a wrong format'.format(
                hb))

    return hbond_dict


def extract_hbonds(hbonds_path):
    hbonds = defaultdict(list)

    with open(str(hbonds_path), 'r') as file:
        # Skip four header lines
        file.readline()
        n_donors = int(file.readline().split()[0])
        n_acceptors = int(file.readline().split()[0])
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

    return hbonds, n_donors, n_acceptors


def hbond_persistance(hbonds_to_track, hbonds):
    persistance_by_hbond = defaultdict(list)
    hbond_persistance = {}
    for PELE_id, _hbonds in hbonds.items():
        current_hbonds = set()
        for hb in set(_hbonds):
            chain, residue, atom = hb.split(':')
            if ((chain, residue) in hbonds_to_track):
                if ((atom in hbonds_to_track[(chain, residue)]) or
                        ('all' in hbonds_to_track[(chain, residue)])):
                    _hb = (chain, residue, tuple(
                        hbonds_to_track[(chain, residue)]))
                    current_hbonds.add(_hb)
                    hbond_persistance[_hb] = hbond_persistance.get(_hb, 0) + 1

        non_current_hbonds = set()
        for p_hb in hbond_persistance.keys():
            if (p_hb not in current_hbonds):
                non_current_hbonds.add(p_hb)

        for hb in non_current_hbonds:
            persistance_by_hbond[hb].append(hbond_persistance.pop(hb))

    non_current_hbonds = set(hbond_persistance.keys())
    for hb in non_current_hbonds:
        persistance_by_hbond[hb].append(hbond_persistance.pop(hb))

    return persistance_by_hbond, len(hbonds)


def get_ligand_rotatable_bonds(lig_rotamers_path):
    counter = 0
    with open(str(lig_rotamers_path), 'r') as lrl:
        for line in lrl:
            line = line.strip()
            if (line.startswith('sidelib')):
                counter += 1
    return counter


def main():
    # Parse args
    PELE_sim_paths, hbonds_relative_path, hbonds, output_path, lig_resname = \
        parse_args()

    hbonds_to_track = prepare_hbond_dict(hbonds)

    print(' - Persistance will be calculated on H bonds :')
    for res, atoms in hbonds_to_track.items():
        print('   - {}:{}:'.format(*res), end='')
        for at in atoms[:-1]:
            print(at, end=',')
        print(atoms[-1])

    all_sim_it = SimIt(PELE_sim_paths)

    for PELE_sim_path in all_sim_it:
        print('')
        print(' - Filtering H bonds from {}'.format(PELE_sim_path))
        hbonds_path = PELE_sim_path.joinpath(hbonds_relative_path)
        lig_rotamers_path = PELE_sim_path.joinpath('DataLocal/' +
                                                   'LigandRotamerLibs/' +
                                                   '{}'.format(lig_resname) +
                                                   '.rot.assign')

        if (not hbonds_path.is_file()):
            print(' - Skipping simulation because hbonds file was ' +
                  'missing')
            continue

        if (not lig_rotamers_path.is_file()):
            print(' - Skipping simulation because ligand rotamer library was' +
                  ' missing')
            continue

        hbonds, n_donors, n_acceptors = extract_hbonds(hbonds_path)

        print(' - Detected {} sets of H bonds'.format(len(hbonds)))

        if (len(hbonds) == 0):
            print(' - Skipping simulation because no H bonds were found')
            continue

        persistance_by_hbond, total_models = hbond_persistance(hbonds_to_track,
                                                               hbonds)

        print(persistance_by_hbond)

        if (total_models == 0):
            print(' - Skipping simulation because no models were found')
            continue

        n_rotamers = get_ligand_rotatable_bonds(lig_rotamers_path)

        print(' - Results:')
        print('   - Ligand rotamers:           {:10d}'.format(n_rotamers))
        print('   - Ligand donors:             {:10d}'.format(n_donors))
        print('   - Ligand acceptors:          {:10d}'.format(n_acceptors))
        print('   - Total models:              {:10d}'.format(total_models))
        if (len(hbonds_to_track) > 0):
            print('   - Maximum persistance by H bond:')
        for (chain, residue), atoms in hbonds_to_track.items():
            print('     - {}:{}:               {:10d}'.format(
                chain, residue,
                max(persistance_by_hbond.get(
                    (chain, residue, tuple(atoms)), [-1, ]))))

        with open(str(PELE_sim_path.joinpath(output_path)), 'w') as f:
            f.write('rotamers;donors;acceptors;models')
            for (chain, residue), atoms in hbonds_to_track.items():
                f.write(';{}:{}:{}'.format(chain, residue, ','.join(atoms)))
            f.write('\n')

            f.write('{};{};{};'.format(n_rotamers, n_donors, n_acceptors))
            f.write('{};'.format(total_models))
            for (chain, residue), atoms in hbonds_to_track.items():
                f.write(';{:d}'.format(max(persistance_by_hbond.get(
                    (chain, residue, tuple(atoms)), [-1, ]))))


if __name__ == "__main__":
    main()
