# -*- coding: utf-8 -*-


# Standard imports
from functools import partial
from multiprocessing import Pool

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


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


def evaluate_logistic_model(X, y, cv_splits=None, random_state=None):
    print(' - Evaluating logistic regression model')
    log_reg = LogisticRegression(random_state=random_state)
    if (cv_splits is not None and cv_splits > 1):
        cv = KFold(n_splits=5)
        results = cross_validate(log_reg, X, y, cv=cv,
                                 return_estimator=True,
                                 scoring='balanced_accuracy')
        score = np.mean(results['test_score'])
    else:
        log_reg.fit(X, y)
        score = metrics.balanced_accuracy_score(y, log_reg.predict(X))

    print('   - Cross-validation results:')
    print('     - Balanced accuracy:', np.mean(results['test_score']))

    return score


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
