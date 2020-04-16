# -*- coding: utf-8 -*-


# Standard imports
from functools import partial
from multiprocessing import Pool

# External imports
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import VotingRegressor
import numpy as np
from matplotlib import pyplot as plt


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


def get_model_from_type(model_type):
    if (model_type.lower() == 'linear'):
        return LinearRegression()
    elif (model_type.lower() == 'lasso'):
        return Lasso()
    elif (model_type.lower() == 'ridge'):
        return Ridge()
    elif (model_type.lower() == 'ensemble'):
        return None
    else:
        raise TypeError('Unknown model type \'{}\''.format(model_type))


class ModelEvaluator(object):
    def __init__(self, model_type, model_params={}, cv_splits=3,
                 cv_scoring='r2'):
        self._model_type = model_type
        self._model = get_model_from_type(model_type)
        self._check_model()
        self._model_params = model_params
        self._cv_splits = cv_splits
        self._cv_scoring = cv_scoring

    @property
    def model_type(self):
        return self._model_type

    @property
    def model(self):
        return self._model

    @property
    def model_params(self):
        return self._model_params

    @property
    def cv_splits(self):
        return self._cv_splits

    @property
    def cv_scoring(self):
        return self._cv_scoring

    def _check_model(self):
        if (self.model is None):
            raise TypeError('Invalid model type \'{}\''.format(
                self.model_type))

    def set_params(self, params):
        self._model_params = params

    def evaluate(self, X, y, verbose=False):
        # Set parameters
        self.model.set_params(**self.model_params)

        # Run cross-validation
        cv = KFold(n_splits=self.cv_splits)
        results = cross_validate(self.model, X, y, cv=cv,
                                 return_estimator=True,
                                 scoring=self.cv_scoring)

        # Print out results
        if (verbose):
            print(' - Cross-validation results with {}:'.format(
                self.model_type))
            print('   - mean {}:'.format(self.cv_scoring),
                  np.mean(results['test_score']))
            print('   - min {}:'.format(self.cv_scoring),
                  np.min(results['test_score']))
            print('   - max {}:'.format(self.cv_scoring),
                  np.max(results['test_score']))

        return results

    def grid_search_evaluation(self, X, y, params_gs, verbose=False):
        # Set parameters
        self.model.set_params(**self.model_params)

        # Run grid-search
        cv = KFold(n_splits=self.cv_splits)
        grid_search = GridSearchCV(self.model, params_gs, cv=cv,
                                   scoring=self.cv_scoring)
        grid_search.fit(X, y)

        # Print out results
        if (verbose):
            print(' - Grid-search results with {}:'.format(
                self.model_type))
            print("   - Best parameters:")
            print("     - {}: {}".format(
                *[(i, j) for i, j in list(
                    grid_search.best_params_.items())][0]))

            mean_train_scores = grid_search.cv_results_['mean_train_score']
            mean_test_scores = grid_search.cv_results_['mean_test_score']

            print("   - Mean scores (train and test):")
            for i, (s1, s2) in enumerate(zip(mean_train_scores,
                                             mean_test_scores)):
                if (i == grid_search.best_index_):
                    print("     * {} {}".format(s1, s2))
                else:
                    print("     - {} {}".format(s1, s2))

        return grid_search.best_params_, grid_search.best_score_

    def plot_linear_relationships(self, X, y, X_labels, y_label, output_path):
        plot_linear_relations(X_labels, X, y, [], [], output_path, y_label)


class ModelGenerator(object):
    def __init__(self, model_type, model_params={}):
        self._model_type = model_type
        self._model = get_model_from_type(model_type)
        self._check_model()
        self._model_params = model_params

    @property
    def model_type(self):
        return self._model_type

    @property
    def model(self):
        return self._model

    @property
    def model_params(self):
        return self._model_params

    def _check_model(self):
        if (self.model is None):
            raise TypeError('Invalid model type \'{}\''.format(
                self.model_type))

    def generate(self, X, y, verbose=False):
        # Set parameters
        self.model.set_params(**self.model_params)

        # Generate model
        self.model.fit(X, y)

        # Get score
        y_pred = self.model.predict(X)
        score = metrics.r2_score(y, y_pred)

        # Print out results
        if (verbose):
            print(' - Model {} generator'.format(self.model_type))
            print('   - r2 score:', score)

        return self.model, score

    def test(self, X, y, verbose=False):
        # Get score
        y_pred = self.model.predict(X)
        score = metrics.r2_score(y, y_pred)

        # Print out results
        if (verbose):
            print(' - Test set results with {}:'.format(self.model_type))
            print('   - r2 score:', score)

        return score

    def predict(self, X, verbose=False):
        # Get score
        y_pred = self.model.predict(X)

        # Print out results
        if (verbose):
            print(' - Model {} predictor'.format(self.model_type))

        return y_pred


class EnsembleGenerator(ModelGenerator):
    def __init__(self, estimators_to_combine, model_params={}):
        super().__init__('Ensemble', model_params)
        self._model = VotingRegressor(estimators=estimators_to_combine)

    def _check_model(self):
        pass


def plot_linear_relations(labels, X_train, X_test, y_train, y_test,
                          output_path, y_label='-pIC50'):
    all_y = np.concatenate((y_train, y_test))

    for i, label in enumerate(labels):
        xs_train = np.array([X[i] for X in X_train])
        xs_test = np.array([X[i] for X in X_test])
        all_xs = np.concatenate((xs_train, xs_test))
        plt.scatter(y_train, xs_train, c='r', label='Training set')
        if (len(xs_test > 0)):
            plt.scatter(y_test, xs_test, c='b', label='Test set')

        lin_reg = LinearRegression()
        lin_reg.fit(xs_train.reshape(-1, 1), y_train)
        for x, y in zip(xs_train, lin_reg.predict(all_xs.reshape(-1, 1))):
            if (x == min(xs_train)):
                min_y = y
            if (x == max(xs_train)):
                max_y = y
        plt.plot((min_y, max_y), (min(xs_train), max(xs_train)), 'r--',
                 linewidth=1, label='Training correlation')
        plt.xlabel(label)
        plt.ylabel(y_label)

        if (len(xs_test > 0)):
            lin_reg = LinearRegression()
            lin_reg.fit(xs_test.reshape(-1, 1), y_test)
            for x, y in zip(xs_test, lin_reg.predict(all_xs.reshape(-1, 1))):
                if (x == min(xs_test)):
                    min_y = y
                if (x == max(xs_test)):
                    max_y = y
            plt.plot((min_y, max_y), (min(xs_test), max(xs_test)), 'b--',
                     linewidth=1, label='Test correlation')
            plt.xlabel(label)
            plt.ylabel(y_label)

            lin_reg = LinearRegression()
            lin_reg.fit(all_xs.reshape(-1, 1), all_y)
            for x, y in zip(all_xs, lin_reg.predict(all_xs.reshape(-1, 1))):
                if (x == min(all_xs)):
                    min_y = y
                if (x == max(all_xs)):
                    max_y = y
            plt.plot((min_y, max_y), (min(all_xs), max(all_xs)), 'k--',
                     linewidth=2, label='Overall correlation')

        plt.legend()

        score = "r2 = {:.3f}".format(
            metrics.r2_score(all_y, lin_reg.predict(all_xs.reshape(-1, 1))))
        props = dict(boxstyle='round', facecolor='grey', alpha=0.25)
        plt.text(min(all_xs), min_y - 0.25, score, fontsize=12,
                 verticalalignment='center', bbox=props)

        plt.savefig(str(output_path.joinpath(
            'linear_relation_{}.png'.format(
                label.replace('/', '').replace(' ', '')))))
        plt.close()
