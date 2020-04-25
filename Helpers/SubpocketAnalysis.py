# -*- coding: utf-8 -*-


# Standard imports
import io
from pathlib import Path

# External imports
import numpy as np

# PELE imports

# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


class Residue(object):
    def __init__(self, chain_id, res_id):
        if (not isinstance(chain_id, int)):
            raise TypeError('Invalid \'chain_id\', it must be an integer')
        self._chain_id = chain_id
        if (not isinstance(res_id, int)):
            raise TypeError('Invalid \'res_id\', it must be an integer')
        self._res_id = res_id

    @property
    def chain_id(self):
        return self._chain_id

    @property
    def res_id(self):
        return self._res_id

    def __repr__(self):
        return "{}:{}".format(self.chain_id, self.res_id)


class ChainConverterForMDTraj(object):
    def __init__(self, path_to_trajectory):
        self._path_to_trajectory = path_to_trajectory
        self._map = {}
        self._build_map()

    @property
    def path_to_trajectory(self):
        return self._path_to_trajectory

    @property
    def map(self):
        return self._map

    def _build_map(self):
        with open(self.path_to_trajectory, 'r') as f:
            self._map = {}
            id_counter = 0
            for line in f:
                if (len(line) > 80):
                    chain = line[21]

                    if (chain not in self.map):
                        self.map[chain.upper()] = id_counter
                        id_counter += 1


class Point(np.ndarray):
    def __new__(cls, coords):
        if (len(coords) != 3):
            raise ValueError('Size of \'coords\' must be 3')

        try:
            for c in coords:
                if (not isinstance(float(c), float)):
                    raise ValueError
        except ValueError:
            raise ValueError('Wrong value found in \'coords\'')

        return np.asarray(coords).view(cls)

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]

    def write_to_PDB(self, path=None, out_file=None, point_id=0):
        if (path is None and out_file is None):
            raise ValueError('Either a \'path\' or a \'out_file\' need to ' +
                             'be supplied')

        if (out_file is not None):
            if (not isinstance(out_file, io.IOBase)):
                raise ValueError('Invalid \'out_file\'')
            out_file.write('ATOM    {:3d}  '.format(point_id) +
                           'CEN BOX A {:3d} '.format(point_id) +
                           '{:>11.3f}{:>8.3f}{:>8.3f}'.format(*self) +
                           '  1.00  0.00\n')

        if (path is not None):
            if (not isinstance(path, str)):
                raise ValueError('Invalid \'path\'')
                path = Path(path)
                if (not path.parent.isdir()):
                    raise ValueError('Path {} does not exist'.format(path))

            with open(str(path), 'w') as f:
                f.write('ATOM    {:3d}  '.format(point_id) +
                        'CEN BOX A {:3d} '.format(point_id) +
                        '{:>11.3f}{:>8.3f}{:>8.3f}'.format(*self) +
                        '  1.00  0.00\n')


class Subpocket(object):
    def __init__(self, list_of_residues):
        self._list_of_residues = list_of_residues

    @property
    def list_of_residues(self):
        return self._list_of_residues

    def get_centroid(self, snapshot):
        coords = []
        for r in self.list_of_residues:
            index = snapshot.top.select('chainid {} '.format(r.chain_id) +
                                        'and residue {} '.format(r.res_id) +
                                        'and name CA')
            coords.append(snapshot.xyz[0][index][0] * 10)

        coords = np.stack(coords)

        return Point(np.mean(coords, axis=0))


def build_residues(residues_list, chain_ids_map=None):
    """
    It builds an array of Residue objects

    Parameters
    ----------
    residues_list : list of (chain_id, residue_id)
        It is a list of residue identifiers that will be used to build
        Residue objectes

    Returns
    -------
    residue_objects : list of Residue objects
        It contains the Residue objectes that have been created with the
        given identifiers
    """

    residue_objectes = []

    for r in residues_list:
        # Check format
        try:
            c_id, r_id = r
        except ValueError:
            raise ValueError('Wrong format in \'residues_list\', only ' +
                             '2-dimensional tuples are allowed: ' +
                             '(chain_id, residue_id)')

        # Check types
        if (not isinstance(c_id, int) and not isinstance(c_id, str)):
            raise TypeError('Wrong \'chain_id\', it can only be a ' +
                            'string or an integer')
        elif (isinstance(c_id, str) and len(c_id) > 1):
            raise TypeError('Wrong \'chain_id\', only single-character ' +
                            'strings are acepted')
        if (not isinstance(r_id, int)):
            raise TypeError('Wrong \'residue_id\', it can only be ' +
                            'an integer')
        if (chain_ids_map is not None):
            if (not isinstance(chain_ids_map, ChainConverterForMDTraj)):
                raise TypeError('Wrong \'chain_ids_map\', it has to be ' +
                                'an instance of ChainCovertedForMDTraj ' +
                                'class')

        # Convert chain_id to the right id
        if (isinstance(c_id, str)):
            if (chain_ids_map is None):
                raise ValueError('When a character is supplied, a ' +
                                 '\'chain_ids_map\' is also required')
            c_id = chain_ids_map.map[c_id.upper()]

        residue_objectes.append(Residue(c_id, r_id))

    return residue_objectes
