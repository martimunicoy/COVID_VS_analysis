# -*- coding: utf-8 -*-

# Standard imports
import argparse as ap
from pathlib import Path
import os
import pickle

# External imports
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")


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


def generate_logistic_model(X_train, y_train, X_test, y_test):
    print(' - Generating logistic regression model')
    log_reg = LogisticRegression(random_state=1993)
    cv = KFold(n_splits=5)
    results = cross_validate(log_reg, X_train, y_train, cv=cv,
                             return_estimator=True,
                             scoring='balanced_accuracy')

    best_model = None
    best_scorer = 0
    for m, s in zip(results['estimator'], results['test_score']):
        if (best_model is None or best_scorer < s):
            best_scorer = s
            best_model = m

    print('   - Cross-validation results:')
    print('     - Balanced accuracy:', np.mean(results['test_score']))

    y_pred = best_model.predict(X_test)

    print("   - Test set results:")
    print("     - Confusion matrix:")
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("        | {} | {} |".format(*confusion_matrix[0]))
    print("        | {} | {} |".format(*confusion_matrix[1]))
    print("     - Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("     - Balanced accuracy:",
          metrics.balanced_accuracy_score(y_test, y_pred))
    print("     - Precision:", metrics.precision_score(y_test, y_pred))
    print("     - Recall:", metrics.recall_score(y_test, y_pred))

    return best_model


def generate_forest_model(X_train, y_train, X_test, y_test):
    print(' - Generating random forest model')

    for_cla = RandomForestClassifier(class_weight='balanced',
                                     random_state=1993)
    cv = KFold(n_splits=5)
    params_rf = {'n_estimators': [50, 100], 'max_depth': [2, 3, 4]}
    for_cla_gs = GridSearchCV(for_cla, params_rf, cv=cv,
                              scoring='balanced_accuracy')
    for_cla_gs.fit(X_train, y_train)

    print('   - Cross-validation results:')
    print('     - Balanced accuracy:', for_cla_gs.best_score_)

    best_model = for_cla_gs.best_estimator_

    y_pred = best_model.predict(X_test)

    print("   - Best parameters:")
    print("     - {}: {}".format(
        *[(i, j) for i, j in list(for_cla_gs.best_params_.items())][0]))
    print("     - {}: {}".format(
        *[(i, j) for i, j in list(for_cla_gs.best_params_.items())][1]))
    print("   - Test set results:")
    print("     - Confusion matrix:")
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("        | {} | {} |".format(*confusion_matrix[0]))
    print("        | {} | {} |".format(*confusion_matrix[1]))
    print("     - Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("     - Balanced accuracy:",
          metrics.balanced_accuracy_score(y_test, y_pred))
    print("     - Precision:", metrics.precision_score(y_test, y_pred))
    print("     - Recall:", metrics.recall_score(y_test, y_pred))

    return best_model


def generate_neighbors_model(X_train, y_train, X_test, y_test):
    print(' - Generating k-nearest neighbors model')
    for_cla = KNeighborsClassifier()
    cv = KFold(n_splits=5)
    params_rf = {'n_neighbors': [1, 2, 3, 4, 5]}
    for_cla_gs = GridSearchCV(for_cla, params_rf, cv=cv,
                              scoring='balanced_accuracy')
    for_cla_gs.fit(X_train, y_train)

    print('   - Cross-validation results:')
    print('     - Balanced accuracy:', for_cla_gs.best_score_)

    best_model = for_cla_gs.best_estimator_

    y_pred = best_model.predict(X_test)

    print("   - Best parameters:")
    print("     - {}: {}".format(
        *[(i, j) for i, j in list(for_cla_gs.best_params_.items())][0]))
    print("   - Test set results:")
    print("     - Confusion matrix:")
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("        | {} | {} |".format(*confusion_matrix[0]))
    print("        | {} | {} |".format(*confusion_matrix[1]))
    print("     - Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("     - Balanced accuracy:",
          metrics.balanced_accuracy_score(y_test, y_pred))
    print("     - Precision:", metrics.precision_score(y_test, y_pred))
    print("     - Recall:", metrics.recall_score(y_test, y_pred))

    return best_model


def generate_ensemble_classifier(models_to_combine,
                                 X_train, y_train, X_test, y_test):
    print(' - Generating ensemble model')

    ensemble = VotingClassifier(estimators=models_to_combine, voting='soft')
    cv = KFold(n_splits=5)
    results = cross_validate(ensemble, X_train, y_train, cv=cv,
                             return_estimator=True,
                             scoring='balanced_accuracy')

    best_model = None
    best_scorer = 0
    for m, s in zip(results['estimator'], results['test_score']):
        if (best_model is None or best_scorer < s):
            best_scorer = s
            best_model = m

    print('   - Cross-validation results:')
    print('     - Balanced accuracy:', np.mean(results['test_score']))

    y_pred = best_model.predict(X_test)

    print("   - Test set results:")
    print("     - Confusion matrix:")
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("        | {} | {} |".format(*confusion_matrix[0]))
    print("        | {} | {} |".format(*confusion_matrix[1]))
    print("     - Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("     - Balanced accuracy:",
          metrics.balanced_accuracy_score(y_test, y_pred))
    print("     - Precision:", metrics.precision_score(y_test, y_pred))
    print("     - Recall:", metrics.recall_score(y_test, y_pred))

    return best_model


def generate_regression_model(X_train, y_train, X_test, y_test):
    print(' - Generating linear regression model')
    lin_reg = LinearRegression()
    cv = KFold(n_splits=5)
    results = cross_validate(lin_reg, X_train, y_train, cv=cv,
                             return_estimator=True,
                             scoring='r2')

    best_model = None
    best_scorer = 0

    for m, s in zip(results['estimator'], results['test_score']):
        if (best_model is None or best_scorer < s):
            best_scorer = s
            best_model = m

    print(X_test)

    y_pred = best_model.predict(X_test)

    print('   - Cross-validation results:')
    print('     - r2:', np.max(results['test_score']))
    print("   - Coefficients:")
    print("     - be / atoms : {}".format(best_model.coef_[0]))
    print("     - rotamers : {}".format(best_model.coef_[1]))
    print("     - internal : {}".format(best_model.coef_[2]))
    print("   - Intercept:")
    print("     - {}".format(best_model.intercept_))
    print("   - Test set results:")
    print("     - r2:", metrics.r2_score(y_test, y_pred))

    return best_model


def generate_lasso_model(X_train, y_train, X_test, y_test):
    print(' - Generating linear lasso model')
    lin_las = Lasso()
    cv = KFold(n_splits=5)
    params_rf = {'alpha': np.arange(0, 1, 0.05)}
    lin_las_gs = GridSearchCV(lin_las, params_rf, cv=cv,
                              scoring='r2')
    lin_las_gs.fit(X_train, y_train)

    best_model = lin_las_gs.best_estimator_

    y_pred = best_model.predict(X_test)

    print('   - Cross-validation results:')
    print('     - r2:', lin_las_gs.best_score_)
    print("   - Best parameters:")
    print("     - {}: {}".format(
        *[(i, j) for i, j in list(lin_las_gs.best_params_.items())][0]))
    print("   - Coefficients:")
    print("     - be / atoms : {}".format(best_model.coef_[0]))
    print("     - rotamers : {}".format(best_model.coef_[1]))
    print("     - internal : {}".format(best_model.coef_[2]))
    print("   - Intercept:")
    print("     - {}".format(best_model.intercept_))
    print("   - Test set results:")
    print("     - r2:", metrics.r2_score(y_test, y_pred))

    return best_model


def generate_ridge_model(X_train, y_train, X_test, y_test):
    print(' - Generating linear ridge model')
    lin_rid = Ridge()
    cv = KFold(n_splits=5)
    results = cross_validate(lin_rid, X_train, y_train, cv=cv,
                             return_estimator=True,
                             scoring='r2')

    best_model = None
    best_scorer = 0

    for m, s in zip(results['estimator'], results['test_score']):
        if (best_model is None or best_scorer < s):
            best_scorer = s
            best_model = m

    y_pred = best_model.predict(X_test)

    print('   - Cross-validation results:')
    print('     - r2:', np.max(results['test_score']))
    print("   - Coefficients:")
    print("     - be / atoms : {}".format(best_model.coef_[0]))
    print("     - rotamers : {}".format(best_model.coef_[1]))
    print("     - internal : {}".format(best_model.coef_[2]))
    print("   - Intercept:")
    print("     - {}".format(best_model.intercept_))
    print("   - Test set results:")
    print("     - r2:", metrics.r2_score(y_test, y_pred))

    return best_model


def generate_ensemble_regressor(models_to_combine,
                                X_train, y_train, X_test, y_test):
    print(' - Generating ensemble model')

    ensemble = VotingRegressor(estimators=models_to_combine)
    cv = KFold(n_splits=5)
    results = cross_validate(ensemble, X_train, y_train, cv=cv,
                             return_estimator=True,
                             scoring='r2')

    best_model = None
    best_scorer = 0

    for m, s in zip(results['estimator'], results['test_score']):
        if (best_model is None or best_scorer < s):
            best_scorer = s
            best_model = m

    y_pred = best_model.predict(X_test)

    print('   - Cross-validation results:')
    print('     - r2:', np.max(results['test_score']))
    print("   - Test set results:")
    print("     - r2:", metrics.r2_score(y_test, y_pred))

    return best_model


def plot_classification(activity_classifier_1, X_test, y_test,
                        title, output_path):
    disp = metrics. plot_confusion_matrix(activity_classifier_1,
                                          X_test, y_test,
                                          cmap=plt.cm.Blues)

    disp.ax_.set_title(title)

    plt.savefig(output_path)
    plt.clf()
    plt.cla()


def plot_regressor(model, X_test, y_test, X_train, y_train, title,
                   output_path, nn=None, min_nn=1):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
    plt.subplots_adjust(left=0.15, bottom=0.3, right=0.95, top=0.9)
    fig.suptitle(title, fontsize=16)

    y_pred = model.predict(X_train)
    ax1.plot(((min(y_train), max(y_train))), ((min(y_pred), max(y_pred))),
             'k--', linewidth=1)
    ax1.scatter(y_train, y_pred, color='red')
    ax1.set_title("Cross-validation results")
    ax1.set_ylabel("Predicted -pIC50")
    ax1.set_xlabel("Experimental -pIC50")

    y_pred = model.predict(X_test)
    ax2.plot(((min(y_test), max(y_test))), ((min(y_pred), max(y_pred))),
             'k--', linewidth=1)
    if (nn is None):
        ax2.scatter(y_test, y_pred, color='red', c=nn)
    else:
        colormap = cm.get_cmap('Wistia', max(nn) - (min_nn - 1))
        norm = Normalize(0, max(nn) - (min_nn - 1))
        colormap.set_under('grey')
        for x, y, n in zip(y_test, y_pred, nn):
            ax2.scatter(x, y, c=[colormap(n - min_nn)], label=n)
        fig.colorbar(cm.ScalarMappable(cmap=colormap, norm=norm), ax=ax2,
                     extend='min')
        X_test, y_test = filter_entries_by_nn(X_test, y_test,
                                              nn,
                                              min_nn=min_nn)
    ax2.set_title("Test set results")
    ax2.set_xlabel("Experimental -pIC50")

    try:
        props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
        text = 'Coefficients:\n'
        text += ' • be / atoms = {:.2f}\n'.format(model.coef_[0])
        text += ' • rotamers = {:.2f}\n'.format(model.coef_[1])
        text += ' • internal = {:.2f}'.format(model.coef_[2])
        # place a text box in upper left in axes coords
        plt.text(0.1, 0.1, text, fontsize=12,
                 verticalalignment='center', bbox=props,
                 transform=fig.transFigure)

        text = 'Intercept:\n'
        text += ' • {:.2f}'.format(model.intercept_)
        plt.text(0.35, 0.1, text, fontsize=12,
                 verticalalignment='center', bbox=props,
                 transform=fig.transFigure)

        text = 'Cross-validation r2:\n'
        text += ' • {:.2f}'.format(model.score(X_train, y_train))
        plt.text(0.55, 0.1, text, fontsize=12,
                 verticalalignment='center', bbox=props,
                 transform=fig.transFigure)

        text = 'Test set r2:\n'
        text += ' • {:.2f}'.format(model.score(X_test, y_test))
        plt.text(0.8, 0.1, text, fontsize=12,
                 verticalalignment='center', bbox=props,
                 transform=fig.transFigure)

    except AttributeError:
        text = 'Cross-validation r2:\n'
        text += ' • {:.2f}'.format(model.score(X_train, y_train))
        plt.text(0.30, 0.1, text, fontsize=12,
                 verticalalignment='center', bbox=props,
                 transform=fig.transFigure)

        text = 'Test set r2:\n'
        text += ' • {:.2f}'.format(model.score(X_test, y_test))
        plt.text(0.75, 0.1, text, fontsize=12,
                 verticalalignment='center', bbox=props,
                 transform=fig.transFigure)

    plt.savefig(output_path)
    plt.clf()
    plt.cla()


def domain_analysis(train_domain):
    print(" - Computing applicability domains")

    # Matrix of descriptor distances between entries
    distances = np.array([cdist([x], train_domain) for x in train_domain])
    distances_sorted = [np.sort(d[0]) for d in distances]

    # Discard 0s, which are distances to the same entry (diagonal
    # elements in the matrix)
    distances_matrix = [d[1:] for d in distances_sorted]

    # Empirical calculation of the smoothing parameter k
    k = int(round(pow(len(train_domain), 1 / 3)))

    # Calculate the mean of first k distances for each sample
    d_means = [np.mean(d[:k][0]) for d in distances_matrix]

    # Calculate quartiles
    Q1 = np.quantile(d_means, .25)
    Q3 = np.quantile(d_means, .75)

    # Calculate reference value
    d_ref = Q3 + 1.5 * (Q3 - Q1)

    # Calculate thresholds
    Dijs = []
    Kis = []
    for distances in distances_matrix:
        allowed_distances = [d for d in distances if d <= d_ref]
        Dijs.append(allowed_distances)
        Kis.append(len(allowed_distances))

    thresholds = []
    for Dij, Ki in zip(Dijs, Kis):
        if (Ki == 0):
            thresholds.append(0)
        else:
            thresholds.append(np.sum(Dij) / Ki)

    thresholds = np.array(thresholds)

    # Convert 0 to minimum value not zero
    thresholds[thresholds == 0] = min(thresholds[thresholds != 0])

    return thresholds


def check_applicability_domain(train_domain, test_domain, thresholds):
    print(" - Checking applicability domains")

    # Matrix of descriptor distances between entries
    distances = np.array([cdist([x], train_domain) for x in test_domain])

    results = []
    for Dijs in distances:
        # get nearest neighbors
        NN = len([j for j, Dij in enumerate(Dijs[0]) if Dij <= thresholds[j]])
        results.append(NN)

    return results


def filter_entries_by_nn(X, y, nn, min_nn):
    f_X = X[[i for i, _ in enumerate(X) if nn[i] >= min_nn]]
    f_y = y[[i for i, _ in enumerate(y) if nn[i] >= min_nn]]

    return f_X, f_y


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
    f_dataset.drop(['toten', 'weight', 'atoms', 'be', 'clusters'], axis=1, inplace=True)

    # Split by activity
    f_dataset.loc[:, 'IC50'] = f_dataset.loc[:, 'IC50'].apply(pd.to_numeric)
    f_dataset['pIC50'] = - np.log10(f_dataset.loc[:, 'IC50'] / 1000000)
    actives, inactives = [x for _, x in f_dataset.groupby(
        [f_dataset['IC50'] < 0])]

    # Convert activity in a boolean
    f_dataset['activity'] = f_dataset['IC50'] > 0
    f_dataset.drop(['IC50'], axis=1, inplace=True)

    """
    # Split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        f_dataset.loc[:, ['be / atoms', 'rotamers', 'internal']].to_numpy(),
        f_dataset['activity'].to_list(),
        train_size=0.75,
        random_state=123,
        stratify=f_dataset['activity'].to_list())

    activity_classifier_1 = generate_logistic_model(X_train, y_train,
                                                    X_test, y_test)

    with open(str(output_path.joinpath('activity_classifier_1.pkl')), 'wb') \
            as handle:
        pickle.dump(activity_classifier_1, handle)

    plot_classification(activity_classifier_1, X_test, y_test,
                        "Activity logistic regression classification",
                        str(output_path.joinpath('activity_classifier_1.png')))

    activity_classifier_2 = generate_forest_model(X_train, y_train,
                                                  X_test, y_test)

    with open(str(output_path.joinpath('activity_classifier_2.pkl')), 'wb') \
            as handle:
        pickle.dump(activity_classifier_2, handle)

    plot_classification(activity_classifier_2, X_test, y_test,
                        "Activity random forest classification",
                        str(output_path.joinpath('activity_classifier_2.png')))

    activity_classifier_3 = generate_neighbors_model(X_train, y_train,
                                                     X_test, y_test)

    with open(str(output_path.joinpath('activity_classifier_3.pkl')), 'wb') \
            as handle:
        pickle.dump(activity_classifier_3, handle)

    plot_classification(activity_classifier_3, X_test, y_test,
                        "Activity k-nearest neighbor classification",
                        str(output_path.joinpath('activity_classifier_3.png')))

    activity_classifier_4 = generate_ensemble_classifier(
        [('Logistic regression', activity_classifier_1),
         ('Random forest', activity_classifier_2,),
         ('k-nearest neighbors', activity_classifier_3)],
        X_train, y_train, X_test, y_test)

    with open(str(output_path.joinpath('activity_classifier_4.pkl')), 'wb') \
            as handle:
        pickle.dump(activity_classifier_4, handle)

    plot_classification(activity_classifier_4, X_test, y_test,
                        "Activity ensemble classification",
                        str(output_path.joinpath('activity_classifier_4.png')))
    """

    # Split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        actives.loc[:, ['be / atoms', 'rotamers', 'internal']].to_numpy(),
        actives['pIC50'].to_numpy(),
        train_size=0.75,
        random_state=123)

    thresholds = domain_analysis(X_train)

    nn_results = check_applicability_domain(X_train, X_test, thresholds)

    min_nn = 0
    f_X_test, f_y_test = filter_entries_by_nn(X_test, y_test,
                                              nn_results,
                                              min_nn=min_nn)
    affinity_predictor_1 = generate_regression_model(X_train, y_train,
                                                     f_X_test, f_y_test)

    with open(str(output_path.joinpath('affinity_predictor_1.pkl')),
              'wb') as handle:
        pickle.dump(affinity_predictor_1, handle)
        pickle.dump(thresholds, handle)

    plot_regressor(affinity_predictor_1, X_test, y_test, X_train, y_train,
                   'Linear regression',
                   str(output_path.joinpath('affinity_predictor_1.png')),
                   nn=nn_results, min_nn=min_nn)

    affinity_predictor_2 = generate_lasso_model(X_train, y_train,
                                                f_X_test, f_y_test)

    with open(str(output_path.joinpath('affinity_predictor_2.pkl')),
              'wb') as handle:
        pickle.dump(affinity_predictor_2, handle)
        pickle.dump(thresholds, handle)

    plot_regressor(affinity_predictor_2, X_test, y_test, X_train, y_train,
                   'Lasso regression',
                   str(output_path.joinpath('affinity_predictor_2.png')),
                   nn=nn_results, min_nn=min_nn)

    affinity_predictor_3 = generate_ridge_model(X_train, y_train,
                                                f_X_test, f_y_test)

    with open(str(output_path.joinpath('affinity_predictor_3.pkl')),
              'wb') as handle:
        pickle.dump(affinity_predictor_3, handle)
        pickle.dump(thresholds, handle)

    plot_regressor(affinity_predictor_3, X_test, y_test, X_train, y_train,
                   'Ridge regression',
                   str(output_path.joinpath('affinity_predictor_3.png')),
                   nn=nn_results, min_nn=min_nn)

    affinity_predictor_4 = generate_ensemble_regressor(
        [('Linear regression', affinity_predictor_1),
         ('Ridge regression', affinity_predictor_3)],
        X_train, y_train, f_X_test, f_y_test)

    with open(str(output_path.joinpath('affinity_predictor_4.pkl')),
              'wb') as handle:
        pickle.dump(affinity_predictor_4, handle)
        pickle.dump(thresholds, handle)

    plot_regressor(affinity_predictor_4, X_test, y_test, X_train, y_train,
                   'Ensemble regression',
                   str(output_path.joinpath('affinity_predictor_4.png')),
                   nn=nn_results, min_nn=min_nn)


if __name__ == "__main__":
    main()
