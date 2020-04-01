# -*- coding: utf-8 -*-

# Standard imports
import argparse as ap
from pathlib import Path
import os
import pickle

# External imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_validate
from matplotlib import pyplot as plt
import numpy as np


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("descriptors_path", metavar="PATH", type=str,
                        help="Path to descriptors")
    parser.add_argument("labels_path", metavar="PATH", type=str,
                        help="Path to labels")
    parser.add_argument("-o", "--output", metavar="PATH", type=str,
                        default="model_out", help="Output path")

    args = parser.parse_args()

    return args.descriptors_path, args.labels_path, args.output


def generate_classification_model(X_train, y_train, X_test, y_test):
    print(' - Generating classification model')
    log_reg = LogisticRegression()
    results = cross_validate(log_reg, X_train, y_train, cv=4,
                             return_estimator=True,
                             scoring='accuracy')

    best_model = None
    best_scorer = 0
    for m, s in zip(results['estimator'], results['test_score']):
        if (best_model is None or best_scorer < s):
            best_scorer = s
            best_model = m

    y_pred = best_model.predict(X_test)

    print("   - Confusion matrix:")
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("      | {} | {} |".format(*confusion_matrix[0]))
    print("      | {} | {} |".format(*confusion_matrix[1]))
    print("   - Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("   - Precision:", metrics.precision_score(y_test, y_pred))
    print("   - Recall:", metrics.recall_score(y_test, y_pred))

    return best_model


def generate_linear_model(X_train, y_train, X_test, y_test):
    print(' - Generating linear model')
    lin_reg = LinearRegression()
    results = cross_validate(lin_reg, X_train, y_train, cv=2,
                             return_estimator=True,
                             scoring='r2')

    best_model = None
    best_scorer = 0

    for m, s in zip(results['estimator'], results['test_score']):
        if (best_model is None or best_scorer < s):
            best_scorer = s
            best_model = m

    y_pred = best_model.predict(X_test)

    print("   - r2:", metrics.r2_score(y_test, y_pred))

    return results


def main():
    descriptors_path, labels_path, output_path = parse_args()

    descriptors_path = Path(descriptors_path)
    labels_path = Path(labels_path)
    output_path = Path(output_path)

    if (not descriptors_path.is_file()):
        print(' - Error: invalid descriptors path \'{}\''.format(
            descriptors_path))

    if (not labels_path.is_file()):
        print(' - Error: invalid dataset path \'{}\''.format(
            labels_path))

    if (not output_path.is_dir()):
        os.mkdir(str(output_path))

    descriptors = pd.read_csv(str(descriptors_path))
    labels = pd.read_csv(str(labels_path))

    dataset = descriptors.merge(labels, left_on='path', right_on='path')

    # Remove unnamed columns
    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

    # Remove non numeric IC50 (exclamation mark!)
    f_dataset = dataset[pd.to_numeric(dataset.IC50,
                                      errors='coerce').notnull()]

    # Calculate BE / number of atoms
    f_dataset['be / atoms'] = f_dataset['be'] / f_dataset['atoms']

    # Drop unnecessary columns
    f_dataset.drop(['toten', 'weight', 'atoms', 'be'], axis=1, inplace=True)

    # Split by activity
    f_dataset.loc[:, 'IC50'] = f_dataset.loc[:, 'IC50'].apply(pd.to_numeric)
    f_dataset['pIC50'] = - np.log10(f_dataset.loc[:, 'IC50'])
    actives, inactives = [x for _, x in f_dataset.groupby(
        [f_dataset['IC50'] < 0])]

    # Convert activity in a boolean
    f_dataset['activity'] = f_dataset['IC50'] > 0
    f_dataset.drop(['IC50'], axis=1, inplace=True)

    # Split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        f_dataset.loc[:, ['be / atoms', 'rotamers', 'internal']].to_numpy(),
        f_dataset['activity'].to_list(),
        train_size=0.75,
        random_state=1993)

    activity_classifier = generate_classification_model(X_train, y_train,
                                                        X_test, y_test)

    with open(str(output_path.joinpath('activity_classifier.pkl')), 'wb') \
            as handle:
        pickle.dump(activity_classifier, handle)

    # Split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        actives.loc[:, ['be / atoms', 'rotamers', 'internal']].to_numpy(),
        actives['pIC50'].to_list(),
        train_size=0.75,
        random_state=1993)

    affinity_predictor = generate_linear_model(X_train, y_train,
                                               X_test, y_test)

    with open(str(output_path.joinpath('affinity_predictor.pkl')), 'wb') \
            as handle:
        pickle.dump(affinity_predictor, handle)


if __name__ == "__main__":
    main()
