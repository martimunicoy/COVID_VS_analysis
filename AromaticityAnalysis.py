# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
from pathlib import Path

# External imports
from rdkit import Chem
import pandas as pd

# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("lig_paths", metavar="PATH", type=str,
                        nargs='*',
                        help="Path to ligand PDB files. They must contain "
                        + "connects")

    parser.add_argument("-o", "--output",
                        metavar="STR", type=str,
                        default="aromaticity.csv")

    parser.add_argument("--alternative_output_path",
                        metavar="PATH", type=str, default=None,
                        help="Alternative path to save output results")

    args = parser.parse_args()

    return args.lig_paths, args.output, args.alternative_output_path


def main():
    lig_paths, output, alternative_output_path = parse_args()

    for lig_path in lig_paths:
        print(' - Analyzing connectivity of {}'.format(lig_path))

        lig_path = Path(lig_path)
        if (not lig_path.is_file()):
            print('   - Invalid file, skipping...')
            continue

        lig = Chem.rdmolfiles.MolFromPDBFile(str(lig_path), removeHs=False)

        data = []
        for atom in lig.GetAtoms():
            data.append((atom.GetPDBResidueInfo().GetResidueName(),
                         atom.GetPDBResidueInfo().GetName(),
                         atom.GetIsAromatic()))

        df = pd.DataFrame(data, columns=['residue', 'atom', 'aromatic'])

        output_path = lig_path.parent
        if (alternative_output_path is not None):
            output_path = Path(alternative_output_path)
        output_path = output_path.joinpath('_'.join(
            (lig_path.name.replace(lig_path.suffix, ''), output)))

        print(' - Aromaticity of {} saved to {}'.format(
            lig_path, output_path))

        df.to_csv(output_path)


if __name__ == "__main__":
    main()
