# -*- coding: utf-8 -*-


# Standard imports
import os
import sys
import io
from pathlib import Path

# External imports
import numpy as np

# Module imports
SCRIPT_PATH = os.path.dirname(__file__)
sys.path.append(os.path.abspath(SCRIPT_PATH))
from Utils import squared_distances


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


# Constants
FOUR_THIRDS_PI = 4.1887902047863905


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
    def __init__(self, list_of_residues, radius=None):
        self._list_of_residues = list_of_residues
        if (radius is None):
            self.fixed_radius = None
        else:
            try:
                self.fixed_radius = float(radius)
            except ValueError:
                raise ValueError('Wrong subpocket radius')

    @property
    def list_of_residues(self):
        return self._list_of_residues

    def get_residue_coords(self, snapshot):
        coords = []
        for r in self.list_of_residues:
            index = snapshot.top.select('chainid {} '.format(r.chain_id) +
                                        'and residue {} '.format(r.res_id) +
                                        'and name CA')
            coords.append(snapshot.xyz[0][index][0] * 10)

        return np.stack(coords)

    def get_centroid(self, snapshot, return_residue_coords=False):
        coords = self.get_residue_coords(snapshot)

        if (return_residue_coords):
            return Point(np.mean(coords, axis=0)), coords
        else:
            return Point(np.mean(coords, axis=0))

    def get_default_radius(self, snapshot, return_centroid=False):
        centroid, coords = self.get_centroid(snapshot, True)

        radius = np.sqrt(np.max(
            [squared_distances(c, centroid) for c in coords]))

        if (return_centroid):
            return radius, centroid
        else:
            return radius

    def get_volume(self, snapshot):
        radius = self.fixed_radius

        if (self.fixed_radius is None):
            radius = self.get_default_radius(snapshot)

        return FOUR_THIRDS_PI * np.power(radius, 3)

    def _calculate_intersections(self, atom_radii, atom_coords, centroid, s_r):
        intersections = {}
        for atom, a_r in atom_radii.items():
            # Calculate bounds
            lower_bound = np.abs(s_r - a_r)
            upper_bound = s_r + a_r

            # Calculate distance d between centers
            coords = atom_coords[atom]
            d = np.linalg.norm(centroid - coords)

            # Minimum radius
            min_radius = np.min((a_r, s_r))

            # Calculate intersections
            if (d >= upper_bound):
                intersections[atom] = 0
            elif (d <= lower_bound):
                intersections[atom] = FOUR_THIRDS_PI * np.power(min_radius, 3)
            else:
                intersections[atom] = np.pi
                intersections[atom] *= np.power(s_r + a_r - d, 2)
                dd = np.power(d, 2)
                da = 2 * d * a_r
                aa = -3 * np.power(a_r, 2)
                ds = 2 * d * s_r
                ss = -3 * np.power(s_r, 2)
                sa = 6 * s_r * a_r
                intersections[atom] *= dd + da + aa + ds + ss + sa
                intersections[atom] /= 12 * d

        return intersections

    def get_intersection(self, snapshot, ligand_resname, centroid=None,
                         radius=None, individual=False):
        if (centroid is None or radius is None):
            if (self.fixed_radius is None):
                radius, centroid = self.get_default_radius(snapshot, True)
            else:
                radius = self.fixed_radius
                centroid = self.get_centroid(snapshot)

        lig = snapshot.topology.select('resname {}'.format(ligand_resname))

        atom_radius = {}
        atom_coords = {}
        for atom_id in lig:
            atom = snapshot.top.atom(atom_id)
            name = atom.name
            atom_radius[name] = atom.element.radius * 10  # In angstroms
            atom_coords[name] = snapshot.xyz[0][atom_id] * 10  # In angstroms

        intersections = self._calculate_intersections(atom_radius, atom_coords,
                                                      centroid, radius)

        if (individual):
            return intersections
        else:
            return np.sum(list(intersections.values()))

    def full_characterize(self, snapshot, ligand_resname):
        if (self.fixed_radius is None):
            radius, centroid = self.get_default_radius(snapshot, True)
        else:
            radius = self.fixed_radius
            centroid = self.get_centroid(snapshot)
        intersection = self.get_intersection(snapshot, ligand_resname,
                                             centroid, radius)

        return centroid, FOUR_THIRDS_PI * np.power(radius, 3), intersection


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
