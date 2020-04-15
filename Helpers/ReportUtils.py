# -*- coding: utf-8 -*-


# Standard imports
from functools import partial
from multiprocessing import Pool

# External imports
import numpy as np


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


def p_extract_metrics(cols, report_path):
    results = []

    if (report_path.is_file()):
        try:
            with open(str(report_path), 'r') as f:
                f.readline()
                for line in f:
                    line.strip()
                    fields = line.split()
                    metrics = []
                    for col in cols:
                        metrics.append(fields[col - 1])
                    results.append(metrics)

        except IndexError:
            print(' - p_extract_ligand_metric Warning: wrong index ' +
                  'supplied for trajectory: \'{}\''.format(report_path))
    else:
        print(' - p_extract_ligand_metric Warning: wrong path to report ' +
              'for trajectory: \'{}\''.format(report_path))

    return results


def extract_metrics(reports, cols, proc_number=None):
    if (proc_number is None and len(reports) == 1):
        return [p_extract_metrics(cols, reports[0]), ]

    parallel_function = partial(p_extract_metrics, cols)

    with Pool(proc_number) as pool:
        results = pool.map(parallel_function,
                           reports)

    return results


def extract_PELE_ids(reports):
    epochs = []
    trajectories = []
    models = []

    for repo in reports:
        with open(str(repo), 'r') as f:
            f.readline()
            for i, line in enumerate(f):
                epochs.append(int(repo.parent.name))
                trajectories.append(
                    int(''.join(filter(str.isdigit, repo.name))))
                models.append(i)

    return epochs, trajectories, models


def get_metric_by_PELE_id(PELE_ids, metrics):
    metric_by_PELE_id = {}
    for e, t, m, metric in zip(*PELE_ids, np.concatenate(metrics)):
        metric_by_PELE_id[(e, t, m)] = metric

    return metric_by_PELE_id
