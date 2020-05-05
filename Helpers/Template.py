# -*- coding: utf-8 -*-


# Standard imports
from collections import defaultdict
from pathlib import Path

# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


class ImpactTemplate(object):
    def __init__(self, path_to_template: str):
        self._path = Path(path_to_template)
        self._parse()

    @property
    def path(self):
        return self._path

    @property
    def ids(self) -> list:
        return self._ids

    @property
    def names(self) -> dict:
        return self._names

    @property
    def parameters(self) -> defaultdict:
        return self._parameters

    @property
    def names_to_ids(self) -> dict:
        return self._names_to_ids

    @property
    def current_section(self) -> str:
        return self._current_section

    def _reset(self):
        self._ids = list()
        self._names = dict()
        self._parameters = defaultdict(dict)
        self._names_to_ids = dict()

    def _create_bidict(self):
        self._names_to_ids = dict([reversed(i) for i in self.names.items()])

    def _parse(self):
        self._reset()

        with open(str(self.path)) as f:
            self._current_section = 'none'
            for line in f:
                if line.startswith('*'):
                    continue
                self._line_parser(line.strip('\n'))

        self._create_bidict()

    def _check_section(self, line: str) -> bool:
        if self._current_section == 'none':
            self._current_section = 'init'
        elif line.startswith('NBON'):
            self._current_section = 'nbon'
        elif line.startswith('BON'):
            self._current_section = 'bon'
        elif line.startswith('THET'):
            self._current_section = 'thet'
        elif line.startswith('PHI'):
            self._current_section = 'phi'
        elif line.startswith('IPHI'):
            self._current_section = 'iphi'
        elif line.startswith('END'):
            self._current_section = 'end'
        else:
            return False
        return True

    def _line_parser(self, line: str):
        if self._check_section(line):
            return

        if self.current_section == 'init':
            self._init_parser(line)
        elif self.current_section == 'nbon':
            self._nbon_parser(line)
        else:
            # Parsers of the other sections are not implemented yet
            pass

    def _init_parser(self, line: str):
        try:
            atom_id = int(line[0:5])
            atom_name = line[21:25]
        except (ValueError, IndexError):
            raise ValueError('Wrong line format:\n{}'.format(line))

        self._ids.append(atom_id)
        self._names[atom_id] = atom_name.strip('_')

    def _nbon_parser(self, line: str):
        try:
            atom_id = int(line[0:6])
            atom_charge = float(line[26:35])
        except (ValueError, IndexError):
            raise ValueError('Wrong line format:\n{}'.format(line))

        self.parameters[atom_id]['charge'] = atom_charge

    def get_id_from_name(self, name: str) -> int:
        return self.names_to_ids[name]

    def get_parameter_by_name(self, name: str, parameter: str) -> float:
        atom_id = self.get_id_from_name(name)

        return self.parameters[atom_id][parameter]
