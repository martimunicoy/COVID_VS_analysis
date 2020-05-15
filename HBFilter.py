# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
from pathlib import Path

# Local imports
from Helpers import SimIt
from Helpers.Hbonds import extract_hbonds

# External imports
import pandas as pd
from ast import literal_eval


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
                        default='hbonds.csv',
                        help="Path to H bonds csv file")

    parser.add_argument("-g1", "--golden_hbonds_1", nargs='*',
                        metavar="C:R[:A1, A2]", type=str, default=[],
                        help="Chain (C), residue (R) [and atoms (A1, A2)] of"
                        + "subset 1 of golden H bonds. Subset 1 contains H "
                        + "bond conditions that must always be fulfilled in "
                        + "the filtering process")

    parser.add_argument("-g2", "--golden_hbonds_2", nargs='*',
                        metavar="C:R[:A1, A2]", type=str, default=[],
                        help="Chain (C), residue (R) [and atoms (A1, A2)] of"
                        + "subset 2 of golden H bonds. Subset 2 contains H "
                        + "bond conditions that only a minimum number of them "
                        + "must be fulfilled in the filtering process. The "
                        + "minimum of required conditions from subset 2 is "
                        + "defined with the minimum_g2_conditions argument")

    parser.add_argument("--minimum_g2_conditions",
                        metavar="N", type=int, default=2,
                        help="Minimum number of subset 2 golden H bonds "
                        + "that must be fulfilled in the filtering process")

    parser.add_argument("-o", "--output_path",
                        metavar="PATH", type=str, default="filter.out")

    parser.add_argument("-l", "--ligand_resname",
                        metavar="LIG", type=str, default='LIG',
                        help="Ligand residue name")

    parser.add_argument("--subpocket_filtering", metavar="STR", type=str,
                        default=None, help="If the relative path to "
                        + "subpockets output file is supplied, subpockets "
                        + "will be filtered by the current H bonds criterion")

    args = parser.parse_args()

    return args.traj_paths, args.hbonds_path, args.golden_hbonds_1, \
        args.golden_hbonds_2, args.minimum_g2_conditions, args.output_path, \
        args.ligand_resname, args.subpocket_filtering


def prepare_golden_dict(golden_hbonds):
    golden_dict = {}
    for hb in golden_hbonds:
        hb_data = hb.split(':')
        if (len(hb_data) == 2):
            golden_dict[tuple(hb_data)] = ['all']
        elif (len(hb_data) == 3):
            golden_dict[tuple(hb_data[0:2])] = hb_data[2].split(',')
        else:
            print('Error: golden H bonds \'{}\' have a wrong format'.format(
                golden_hbonds))

    return golden_dict


def hbond_fulfillment(data, golden_hbonds_1, golden_hbonds_2,
                      minimum_g2_conditions):
    hbonds = data.hbonds.to_list()
    total_fulfillments = 0
    fulfillments_by_g1_hbond = {}
    fulfillments_by_g2_hbond = {}
    mask = []
    for _hbonds in hbonds:
        g1_matchs = 0
        g2_matchs = 0
        for hb in set(_hbonds):
            chain, residue, atom = hb.split(':')
            if ((chain, residue) in golden_hbonds_1):
                if ((atom in golden_hbonds_1[(chain, residue)])
                        or ('all' in golden_hbonds_1[(chain, residue)])):
                    g1_matchs += 1
                    _hb = (chain, residue, tuple(
                        golden_hbonds_1[(chain, residue)]))
                    fulfillments_by_g1_hbond[_hb] = \
                        fulfillments_by_g1_hbond.get(_hb, 0) + 1

            if ((chain, residue) in golden_hbonds_2):
                if ((atom in golden_hbonds_2[(chain, residue)])
                        or ('all' in golden_hbonds_2[(chain, residue)])):
                    g2_matchs += 1
                    _hb = (chain, residue, tuple(
                        golden_hbonds_2[(chain, residue)]))
                    fulfillments_by_g2_hbond[_hb] = \
                        fulfillments_by_g2_hbond.get(_hb, 0) + 1

        if ((g1_matchs == len(golden_hbonds_1))
                and (g2_matchs >= minimum_g2_conditions)):
            total_fulfillments += 1
            accepted = True
        else:
            accepted = False

        mask.append(accepted)

    return total_fulfillments, len(hbonds), fulfillments_by_g1_hbond, \
        fulfillments_by_g2_hbond, data[mask]


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
    PELE_sim_paths, hbonds_relative_path, golden_hbonds_1, golden_hbonds_2, \
        minimum_g2_conditions, output_path, lig_resname, \
        subpockets_relative_filtering = parse_args()

    golden_hbonds_1 = prepare_golden_dict(golden_hbonds_1)
    golden_hbonds_2 = prepare_golden_dict(golden_hbonds_2)

    print(' - Golden H bonds set 1:')
    for res, atoms in golden_hbonds_1.items():
        print('   - {}:{}:'.format(*res), end='')
        for at in atoms[: -1]:
            print(at, end=',')
        print(atoms[-1])
    print(' - Golden H bonds set 2 ({} '.format(minimum_g2_conditions)
          + 'of them need to be fulfilled):')
    for res, atoms in golden_hbonds_2.items():
        print('   - {}:{}:'.format(*res), end='')
        for at in atoms[:-1]:
            print(at, end=',')
        print(atoms[-1])

    all_sim_it = SimIt(PELE_sim_paths)

    for PELE_sim_path in all_sim_it:
        print('')
        print(' - Filtering H bonds from {}'.format(PELE_sim_path))
        hbonds_path = PELE_sim_path.joinpath(hbonds_relative_path)
        lig_rotamers_path = PELE_sim_path.joinpath('DataLocal/'
                                                   + 'LigandRotamerLibs/'
                                                   + '{}'.format(lig_resname)
                                                   + '.rot.assign')

        if (not hbonds_path.is_file()):
            print(' - Skipping simulation because hbonds file was '
                  + 'missing')
            continue

        if (not lig_rotamers_path.is_file()):
            print(' - Skipping simulation because ligand rotamer library was'
                  + ' missing')
            continue

        hbonds, n_donors, n_acceptors = extract_hbonds(hbonds_path)

        print(' - Detected {} sets of H bonds'.format(len(hbonds)))

        if (len(hbonds) == 0):
            print(' - Skipping simulation because no H bonds were found')
            continue

        total_fulfillments, total_models, fulfillments_by_g1_hbond, \
            fulfillments_by_g2_hbond, f_hbonds = \
            hbond_fulfillment(hbonds, golden_hbonds_1, golden_hbonds_2,
                              minimum_g2_conditions)

        if (total_models == 0):
            print(' - Skipping simulation because no models were found')
            continue

        n_rotamers = get_ligand_rotatable_bonds(lig_rotamers_path)
        ratio = total_fulfillments / total_models

        print(' - Results:')
        print('   - Ligand rotamers:           {:10d}'.format(n_rotamers))
        print('   - Ligand donors:             {:10d}'.format(n_donors))
        print('   - Ligand acceptors:          {:10d}'.format(n_acceptors))
        print('   - Total models:              {:10d}'.format(total_models))
        print('   - Total H bond fulfillments: {:10d}'.format(
            total_fulfillments))
        print('   - Fulfillment ratio:         {:10.4f}'.format(ratio))
        if (len(golden_hbonds_1) > 0):
            print('   - Fulfillments ratio by g1 hbond:')
        for (chain, residue), atoms in golden_hbonds_1.items():
            print('     - {}:{}:{}: {:.4f}'.format(
                chain, residue, atoms,
                fulfillments_by_g1_hbond.get(
                    (chain, residue, tuple(atoms)), 0) / total_models))
        if (len(golden_hbonds_2) > 0):
            print('   - Fulfillments ratio by g2 hbond:')
        for (chain, residue), atoms in golden_hbonds_2.items():
            print('     - {}:{}:{}: {:.4f}'.format(
                chain, residue, ','.join(atoms),
                fulfillments_by_g2_hbond.get(
                    (chain, residue, tuple(atoms)), 0) / total_models))

        with open(str(PELE_sim_path.joinpath(output_path)), 'w') as f:
            f.write('rotamers;donors;acceptors;models;fulfillments;ratio')
            for (chain, residue), atoms in golden_hbonds_1.items():
                f.write(';{}:{}:{}'.format(chain, residue, ','.join(atoms)))
            for (chain, residue), atoms in golden_hbonds_2.items():
                f.write(';{}:{}:{}'.format(chain, residue, ','.join(atoms)))
            f.write('\n')

            f.write('{};{};{};'.format(n_rotamers, n_donors, n_acceptors))
            f.write('{};{};{:.4f}'.format(total_models, total_fulfillments,
                                          ratio))
            for (chain, residue), atoms in golden_hbonds_1.items():
                f.write(';{:.4f}'.format(fulfillments_by_g1_hbond.get(
                    (chain, residue, tuple(atoms)), 0) / total_models))
            for (chain, residue), atoms in golden_hbonds_2.items():
                f.write(';{:.4f}'.format(fulfillments_by_g2_hbond.get(
                    (chain, residue, tuple(atoms)), 0) / total_models))

        if subpockets_relative_filtering is not None:
            subpockets_filtering = Path(subpockets_relative_filtering)
            if (not subpockets_filtering.is_file()):
                print(' - Skipping subpocket filtering because subpocket '
                      + 'output file was missing')
                continue

            subpockets_path = PELE_sim_path.joinpath(subpockets_filtering)

            subpockets_data = pd.read_csv(str(subpockets_path))

            f_subpockets = subpockets_data.merge(
                f_hbonds, on=['epoch', 'trajectory', 'model'])

            f_subpockets = pd.merge(f_hbonds, subpockets_data.drop_duplicates(
                subset=['epoch', 'trajectory', 'model']))

            f_subpockets_path = subpockets_path.parent.joinpath(
                '{}_hbonds_filter.csv'.format(
                    subpockets_path.name.replace(subpockets_path.suffix, '')))

            f_subpockets.to_csv(str(f_subpockets_path))


if __name__ == "__main__":
    main()
