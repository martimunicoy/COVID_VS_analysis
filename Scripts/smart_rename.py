# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
from pathlib import Path


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


def parse_args():
    parser = ap.ArgumentParser()

    parser.add_argument("folders", metavar="PATH", type=str,
                        nargs='*',
                        help="Set of folders that will be renamed")

    parser.add_argument("-d", "--delimiter", metavar="STR", type=str,
                        help="Delimiter that want to be splitted by",
                        default='_')

    parser.add_argument("-f", "--field", metavar="INT[INT,INT]", type=str,
                        help="Field or set of fields that will be kept",
                        default='-1')

    args = parser.parse_args()

    return args.folders, args.delimiter, args.field


def main():
    folders, delimiter, field = parse_args()

    fields = []
    for f in field.split(','):
        fields.append(int(f) - 1)

    if (len(fields) == 0):
        raise ValueError('No field was selected')

    for folder in folders:
        folder = Path(folder)

        if (not folder.is_dir()):
            print(' - Skipping folder {} because it does not exist'.format(
                folder))
            continue

        name = folder.name
        try:
            new_name = []
            for f in fields:
                new_name.append(name.split(delimiter)[f])
            new_name = '_'.join(new_name)
        except IndexError:
            print(' - Skipping folder {} because chosen '.format(folder) +
                  'fields were out of range')
            continue

        folder.rename(new_name)


if __name__ == "__main__":
    main()
