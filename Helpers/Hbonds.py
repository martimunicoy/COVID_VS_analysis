# -*- coding: utf-8 -*-


# Standard imports
from pathlib import Path
from typing import Tuple, List, Optional, Union, FrozenSet, Set

# External imports
import pandas as pd


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


class HBondLinker(object):
    def __init__(self, chain: str, residue: str,
                 atoms: Union[List[str], Tuple[str, ...], None] = None):
        self._chain = chain
        self._residue = residue
        self.set_atoms(atoms)

    @property
    def chain(self) -> str:
        return self._chain

    @property
    def residue(self) -> str:
        return self._residue

    @property
    def atoms(self) -> FrozenSet[str]:
        return self._atoms

    def __str__(self) -> str:
        return 'HBondLinker ' + self.__repr__()

    def __repr__(self) -> str:
        return '{}:{}:'.format(self.chain, self.residue) \
            + ','.join(self.atoms)

    def __eq__(self, other) -> bool:
        return (self.chain, self.residue, self.atoms) == \
            (other.chain, other.residue, other.atoms)

    def __hash__(self) -> int:
        return hash((self.chain, self.residue, self.atoms))

    def set_chain(self, chain: str):
        self._chain = chain

    def set_residue(self, residue: str):
        self._residue = residue

    def set_atoms(self,
                  atoms: Union[List[str], Tuple[str, ...], None] = None):
        if atoms is None:
            self._atoms = frozenset(('*', ))
        else:
            self._atoms = frozenset(atoms)

    def match_with(self, other: 'HBondLinker') -> bool:
        if (self.chain, self.residue) == (other.chain, other.residue):
            if '*' in self.atoms:
                return True
            if '*' in other.atoms:
                return True
            if len(self.atoms.intersection(other.atoms)) > 0:
                return True
        return False

    def match_with_any(self, others: List['HBondLinker']
                       ) -> Tuple[bool, Optional['HBondLinker']]:
        for other in others:
            if self.match_with(other):
                return True, other
        return False, None


def get_hbond_linkers(h_bonds: List[str]) -> List[HBondLinker]:
    hb_linker_list = []  # type: List[HBondLinker]
    for hb in h_bonds:
        # hb has the following format CHAIN:RESIDUE[:ATOM1,ATOM2]
        hb_data = hb.split(':')
        if len(hb_data) not in (2, 3):
            raise ValueError('Error: H bonds '
                             + '\'{}\' '.format(h_bonds)
                             + 'have a wrong format')
        # In case there are no atoms specified, take them all
        if len(hb_data) == 2:
            chain, residue = hb_data[:2]
            atoms = None
        # In case atoms are specified
        elif len(hb_data) == 3:
            chain, residue = hb_data[:2]
            atoms = tuple(hb_data[2].split(','))
        # Otherwise, wrong format
        else:
            raise ValueError('Error: H bonds '
                             + '\'{}\' '.format(h_bonds)
                             + 'have a wrong format')
        hb_linker = HBondLinker(chain, residue, atoms)

        hb_linker_list.append(hb_linker)

    return hb_linker_list


def check_hbonds_linkers(hbond_linkers: List[HBondLinker]):
    hb_linkers_found = set()  # type: Set[HBondLinker]
    for hb_linker in hbond_linkers:
        if hb_linker in hb_linkers_found:
            raise ValueError('{} defined twice'.format(hb_linker))

        for hb_linker_found in hb_linkers_found:
            if (hb_linker_found.chain == hb_linker.chain
                    and hb_linker_found.residue == hb_linker.residue):
                print(hb_linker_found.atoms)
                print(hb_linker.atoms)
            if (hb_linker_found.chain == hb_linker.chain
                    and hb_linker_found.residue == hb_linker.residue
                    and ('*' in hb_linker_found.atoms
                         or '*' in hb_linker.atoms
                         or len(hb_linker_found.atoms.intersection(
                            hb_linker.atoms)) != 0)):
                raise ValueError('{} and {} share some atoms'.format(
                    hb_linker_found, hb_linker))

        hb_linkers_found.add(hb_linker)


def print_hbonds(hbond_linkers: List[HBondLinker]):
    for hbl in hbond_linkers:
        print('   - {}:{}:{}'.format(hbl.chain, hbl.residue,
                                     ','.join(hbl.atoms)))


def build_linkers(raw_linkers):
    hb_bunch = raw_linkers.strip('\'\"')
    hbs = hb_bunch.split()
    hbs = [i.strip(',[]') for i in hbs]
    return get_hbond_linkers(hbs)


def extract_hbond_linkers(hbonds_path: Path) -> Tuple[pd.DataFrame, int, int]:
    output_info_path = Path(str(hbonds_path).replace(hbonds_path.suffix, '')
                            + '.info')

    if (output_info_path.is_file()):
        with open(str(output_info_path), 'r') as file:
            # Skip four header lines
            file.readline()
            n_donors = int(file.readline().split()[0])
            n_acceptors = int(file.readline().split()[0])
            file.readline()
    else:
        print('   - Warning: H bonds info output file not found')
        n_donors = -1
        n_acceptors = -1

    data = pd.read_csv(str(hbonds_path))

    data.hbonds = data.hbonds.apply(build_linkers)

    return data, n_donors, n_acceptors
