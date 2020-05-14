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

    parser.add_argument("--prefix", nargs='?', action='append',
                        metavar="C:R", type=str, default=[],
                        help="Prefix pattern to append to files")

    parser.add_argument("--suffix", nargs='?', action='append',
                        metavar="C:R", type=str, default=[],
                        help="Suffix pattern to append to files")

    parser.add_argument("--cut", nargs='?', action='append',
                        metavar="C:R", type=int, default=[],
                        help="Field to be cut off from files")

    parser.add_argument("-d", "--delimiter", metavar="STR", type=str,
                        help="Delimiter that want to be splitted by",
                        default='_')

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

    return args.folders, args.prefix, args.suffix, args.cut, \
        args.delimiter, args.debug, args.force


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


def get_new_path(folder, prefixes, suffixes, cut, delimiter):
    name = folder.name

    new_name = cut_name(name, cut, delimiter)
    if (new_name == ''):
        print(' - Skipping folder {} because chosen '.format(folder)
              + 'cut indexes erased completely the folder name')
        return None

    new_name = append_prefixes(new_name, prefixes, delimiter)
    new_name = append_suffixes(new_name, suffixes, delimiter)

    if new_name == name:
        print(' - Skipping folder {} because nothing '.format(folder)
              + 'has to be done')
        return None

    return new_name


def cut_name(name, cut, delimiter):
    try:
        fields = name.split(delimiter)
    except ValueError:
        fields = [name, ]

    accepted_fields = []
    for i, f in enumerate(fields):
        if i + 1 not in cut:
            accepted_fields.append(f)
    return '{}'.format(delimiter).join(accepted_fields)


def append_prefixes(name, prefixes, delimiter):
    for prefix in prefixes:
        name = prefix + delimiter + name
    return name


def append_suffixes(name, suffixes, delimiter):
    for suffix in suffixes:
        name += delimiter + suffix
    return name


def main():
    folders, prefixes, suffixes, cut, delimiter, debug, force = \
        parse_args()

    for folder in folders:
        folder = Path(folder)

        if (not folder.is_dir()):
            print(' - Skipping folder {} because it does not exist'.format(
                folder))
            continue

        new_name = get_new_path(folder, prefixes, suffixes, cut, delimiter)

        if new_name is None:
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
