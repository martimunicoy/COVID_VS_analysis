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

    parser.add_argument("-g1", "--golden_hbonds_1", nargs='*',
                        metavar="C:R[:A1, A2]", type=str, default=[],
                        help="Chain (C), residue (R) [and atoms (A1, A2)] of" +
                        "subset 1 of golden H bonds. Subset 1 contains H " +
                        "bond conditions that must always be fulfilled in " +
                        "the filtering process")

    parser.add_argument("-g2", "--golden_hbonds_2", nargs='*',
                        metavar="C:R[:A1, A2]", type=str, default=[],
                        help="Chain (C), residue (R) [and atoms (A1, A2)] of" +
                        "subset 2 of golden H bonds. Subset 2 contains H " +
                        "bond conditions that only a minimum number of them " +
                        "must be fulfilled in the filtering process. The " +
                        "minimum of required conditions from subset 2 is " +
                        "defined with the minimum_g2_conditions argument")

    parser.add_argument("--minimum_g2_conditions",
                        metavar="N", type=int, default=2,
                        help="Minimum number of subset 2 golden H bonds " +
                        "that must be fulfilled in the filtering process")

    parser.add_argument("-o", "--output_path",
                        metavar="PATH", type=str, default="filter.out")

    args = parser.parse_args()

    return args.traj_paths, args.hbonds_path, args.golden_hbonds_1, \
        args.golden_hbonds_2, args.minimum_g2_conditions, args.output_path


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


def hbond_fulfillment(hbonds, golden_hbonds_1, golden_hbonds_2,
                      minimum_g2_conditions):
    total_models = 0
    total_fulfillments = 0
    fulfillments_by_g1_hbond = {}
    fulfillments_by_g2_hbond = {}
    for PELE_id, _hbonds in hbonds.items():
        g1_matchs = 0
        g2_matchs = 0
        for hb in set(_hbonds):
            chain, residue, atom = hb.split(':')
            if ((chain, residue) in golden_hbonds_1):
                if ((atom in golden_hbonds_1[(chain, residue)]) or
                        ('all' in golden_hbonds_1[(chain, residue)])):
                    g1_matchs += 1
                    _hb = (chain, residue, tuple(
                        golden_hbonds_1[(chain, residue)]))
                    fulfillments_by_g1_hbond[_hb] = \
                        fulfillments_by_g1_hbond.get(_hb, 0) + 1

            if ((chain, residue) in golden_hbonds_2):
                if ((atom in golden_hbonds_2[(chain, residue)]) or
                        ('all' in golden_hbonds_2[(chain, residue)])):
                    g2_matchs += 1
                    _hb = (chain, residue, tuple(
                        golden_hbonds_2[(chain, residue)]))
                    fulfillments_by_g2_hbond[_hb] = \
                        fulfillments_by_g2_hbond.get(_hb, 0) + 1

        if ((g1_matchs == len(golden_hbonds_1)) and
                (g2_matchs >= minimum_g2_conditions)):
            total_fulfillments += 1

        total_models += 1

    return total_fulfillments, total_models, fulfillments_by_g1_hbond, \
        fulfillments_by_g2_hbond


def main():
    # Parse args
    PELE_sim_paths, hbonds_relative_path, golden_hbonds_1, golden_hbonds_2, \
        minimum_g2_conditions, output_path = parse_args()

    golden_hbonds_1 = prepare_golden_dict(golden_hbonds_1)
    golden_hbonds_2 = prepare_golden_dict(golden_hbonds_2)

    print(' - Golden H bonds set 1:')
    for res, atoms in golden_hbonds_1.items():
        print('   - {}:{}:'.format(*res), end='')
        for at in atoms[:-1]:
            print(at, end=',')
        print(atoms[-1])
    print(' - Golden H bonds set 2 ({} '.format(minimum_g2_conditions) +
          'of them need to be fulfilled):')
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

        if (not hbonds_path.is_file()):
            print(' - Skipping simulation because hbonds file was ' +
                  'missing')
            continue

        hbonds, n_donors, n_acceptors = extract_hbonds(hbonds_path)

        print(' - Detected {} sets of H bonds'.format(len(hbonds)))

        if (len(hbonds) == 0):
            print(' - Skipping simulation because no H bonds were found')
            continue

        total_fulfillments, total_models, fulfillments_by_g1_hbond, \
            fulfillments_by_g2_hbond = hbond_fulfillment(hbonds,
                                                         golden_hbonds_1,
                                                         golden_hbonds_2,
                                                         minimum_g2_conditions)

        if (total_models == 0):
            print(' - Skipping simulation because no models were found')
            continue

        ratio = total_fulfillments / total_models

        print(' - Results:')
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
            f.write('donors;accetors;models;fulfillments;ratio')
            for (chain, residue), atoms in golden_hbonds_1.items():
                f.write(';{}:{}:{}'.format(chain, residue, ','.join(atoms)))
            for (chain, residue), atoms in golden_hbonds_2.items():
                f.write(';{}:{}:{}'.format(chain, residue, ','.join(atoms)))
            f.write('\n')

            f.write('{};{};'.format(n_donors, n_acceptors))
            f.write('{};{};{:.4f}'.format(total_models, total_fulfillments,
                                          ratio))
            for (chain, residue), atoms in golden_hbonds_1.items():
                f.write(';{:.4f}'.format(fulfillments_by_g1_hbond.get(
                    (chain, residue, tuple(atoms)), 0) / total_models))
            for (chain, residue), atoms in golden_hbonds_2.items():
                f.write(';{:.4f}'.format(fulfillments_by_g2_hbond.get(
                    (chain, residue, tuple(atoms)), 0) / total_models))


if __name__ == "__main__":
    main()
