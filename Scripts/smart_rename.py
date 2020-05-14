# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
from pathlib import Path
import sys
from shutil import move

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

    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        help='Debug the behavior of the script without doing'
                        + 'any real modification')

    parser.add_argument('--force',
                        dest='force',
                        action='store_true',
                        help='Force action without asking first')

    parser.set_defaults(debug=False)
    parser.set_defaults(force=False)

    args = parser.parse_args()

    return args.folders, args.delimiter, args.field, args.debug, args.force


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def main():
    folders, delimiter, field, debug, force = parse_args()

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
            print(' - Skipping folder {} because chosen '.format(folder)
                  + 'fields were out of range')
            continue

        new_path = folder.parent.joinpath(new_name)

        if new_path.is_dir():
            action = 'Move'
        else:
            action = 'Rename'

        if (debug):
            print(' - Folder {} would be {}d to {}'.format(
                folder, action.lower(), new_name))
        else:
            if not force:
                if not query_yes_no('{} folder {} to {}?'.format(
                        action, folder, new_name)):
                    return

            print(' - {}ing {} to {}'.format(action[:-1], folder, new_name))
            if action == 'Move':
                for f in folder.iterdir():
                    move(str(f), str(new_path))
                folder.rmdir()

            elif action == 'Rename':
                folder.rename(new_name)


if __name__ == "__main__":
    main()
