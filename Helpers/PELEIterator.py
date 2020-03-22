# -*- coding: utf-8 -*-


# Standard imports
import glob
from pathlib import Path


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


# Make it not callable
class PELEIterator(object):
    def __init__(self, paths=None):
        self._paths = []
        if (paths is not None):
            self.set_paths(paths)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if (self._index < len(self.paths)):
            self._index += 1
            return self.paths[self._index - 1]
        else:
            raise StopIteration()

    @property
    def paths(self):
        return self._paths

    def set_paths(self, paths):
        paths_list = []
        if (type(paths) == list):
            for p in paths:
                paths_list += glob.glob(str(p))
        else:
            paths_list = glob.glob(str(paths))

        paths_list = [Path(i) for i in paths_list]

        if (self._checker(paths_list)):
            self._paths = []
            for p in paths_list:
                self._paths.append(Path(p))
        else:
            raise TypeError('Invalid PELE path type')


class SimIt(PELEIterator):
    def __init__(self, paths=None):
        super().__init__(paths)
        self._traj_it = TrajIt()
        self._repo_it = RepoIt()

    @property
    def traj_it(self):
        return self._traj_it

    @property
    def repo_it(self):
        return self._repo_it

    def build_traj_it(self, relative_output_path='output',
                      traj_name='trajectory', traj_ext='pdb'):
        paths_list = []
        if (traj_ext != ''):
            traj_ext = '.{}'.format(traj_ext)
        for p in self:
            tps = p.joinpath(relative_output_path).glob(
                '[0-9]*/{}_*{}'.format(traj_name, traj_ext))
            for tp in tps:
                paths_list.append(str(tp))

        self._traj_it.set_paths(paths_list)

    def build_repo_it(self, relative_output_path='output',
                      report_name='report', report_ext=''):
        paths_list = []
        if (report_ext != ''):
            report_ext = '.{}'.format(report_ext)
        for p in self:
            tps = p.joinpath(relative_output_path).glob(
                '[0-9]*/{}_*{}'.format(report_name, report_ext))
            for tp in tps:
                paths_list.append(str(tp))

        self._repo_it.set_paths(paths_list)

    def _checker(self, paths_list):
        return all([p.is_dir() for p in paths_list])


class TrajIt(PELEIterator):
    def __init__(self, paths=None):
        super().__init__(paths)

    def _checker(self, paths_list):
        return all([p.is_file() for p in paths_list])


class RepoIt(PELEIterator):
    def __init__(self, paths=None):
        super().__init__(paths)

    def _checker(self, paths_list):
        return all([p.is_file() for p in paths_list])
