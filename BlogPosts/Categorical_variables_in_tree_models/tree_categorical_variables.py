# coding: utf-8

from collections import defaultdict
from operator import itemgetter
import os
import random
import string
import sys
import time

import h2o
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from h2o.estimators.random_forest import H2ORandomForestEstimator


h2o.init()
h2o.no_progress()


def generate_dataset(
        num_x,
        n_samples,
        n_levels=100):
    """
    Generates a dataset such that columns c and z are perfectly predictive of y,
    but with additional features x_i that are weakly predictive
    (and co-correlated).

    Parameters
    ----------
    num_x : int

    n_samples : int

    n_levels : int
        This is the total number of levels, so this value is halved for each
        set (positive and negative).

    Returns
    -------
    df : pd.DataFrame
    """
    X, y = generate_continuous_data_and_targets(num_x, n_samples)
    c, z = make_c_and_z_based_on_y(y, n_levels)
    df_cat = pd.DataFrame(
        X, columns=['x{}'.format(i) for i in range(X.shape[1])])
    for col, name in zip([z, c], ['z', 'c']):
        df_cat = pd.DataFrame(col, columns=[name]).join(df_cat)
    df_ohe = pd.get_dummies(df_cat, 'c')
    df_cat = pd.DataFrame(y, columns=['y']).join(df_cat)
    df_ohe = pd.DataFrame(y, columns=['y']).join(df_ohe)
    return df_cat, df_ohe


def generate_continuous_data_and_targets(
        n_dim,
        n_samples,
        mixing_factor=0.025):
    """
    Generates a multivariate Gaussian-distributed dataset and a response
    variable that is conditioned on a weighted sum of the data.

    Parameters
    ----------
    n_dim : int

    n_samples : int

    mixing_factor : float
        'Squashes' the weighted sum into the linear regime of a sigmoid.
        Smaller numbers squash closer to 0.5.

    Returns
    -------
    X : np.array
        (n_samples, n_dim)

    y : np.array
        (n_samples, )
    """
    cov = generate_positive_semi_definite_matrix(n_dim)
    X = np.random.multivariate_normal(
            mean=np.zeros(n_dim),
            cov=cov,
            size=n_samples)
    weights = np.random.randn(n_dim)
    y_probs = sigmoid(mixing_factor * np.dot(X, weights))
    y = np.random.binomial(1, p=y_probs)
    return X, y


def generate_positive_semi_definite_matrix(n_dim):
    """
    Creates a positive semi-definite matrix.

    Parameters
    ----------
    n_dim : int

    Returns
    -------
    np.array : (n_dim, n_dim)
    """
    cov = np.random.randn(n_dim, n_dim)
    return np.dot(cov, cov.T)


def sigmoid(x):
    """
    Computes sigmoid(x) for some activation x.

    Parameters
    ----------
    x : float

    Returns
    -------
    sigmoid(x) : float
    """
    return 1 / (1 + np.exp(-x))


def make_C(n_levels=100):
    """
    Create two lists, one starting with 'A' and one starting with 'B',
    and each with n/2 levels.

    Parameters
    ----------
    n_levels : int
        Cardinality of C

    Returns
    -------
    Cpos : list of strings

    Cneg : list of strings
    """
    suffixes = ["{}{}".format(i, j) for i in string.ascii_lowercase for j
                in string.ascii_lowercase]
    return ["A{}".format(s) for s in suffixes][:int(n_levels/2)], \
           ["B{}".format(s) for s in suffixes][:int(n_levels/2)]


def make_c_and_z_based_on_y(y_vals, n_levels, z_pivot=10):
    """
    Builds a categorical variable c and continuous variable z such that
    y is perfectly predictable from c and z, with y = 1 iff c takes a value
    from a positive set OR z > z_pivot.

    Parameters
    ----------
    y_vals : np.array

    n_levels : int
        Cardinality of the categorical variable, c.

    z_pivot : float
        Mean of z.

    Returns
    -------
    c : np.array

    z : np.array
    """
    z = np.random.normal(loc=z_pivot, scale=5, size=2 * len(y_vals))
    z_pos, z_neg = z[z > z_pivot], z[z <= z_pivot]
    c_pos, c_neg = make_C(n_levels)
    c, z = list(), list()
    for y in y_vals:
        coin = np.random.binomial(1, 0.5)
        if y and coin:
            c.append(random.choice(c_pos + c_neg))
            z.append(random.choice(z_pos))
        elif y and not coin:
            c.append(random.choice(c_pos))
            z.append(random.choice(z_neg))
        else:
            c.append(random.choice(c_neg))
            z.append(random.choice(z_neg))
    return np.array(c), z


def get_feature_names(df, include_c):
    """
    Returns a list of feature names from a dataframe, optionally excluding
    categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
    include_c : bool

    Returns
    -------
    names : list of strings
    """
    names = [f for f in df.columns if not f.startswith('y')]
    if not include_c:
        names = [f for f in names if not f.startswith('c')]
    return names


class H2ODecisionTree:
    """
    Simple class that overloads an H2ORandomForestEstimator to mimic a
    decision tree classifier. Only train, predict and varimp are implemented.
    """
    def __init__(self):
        self.model = None

    def train(self, x, y, training_frame):
        self.model = H2ORandomForestEstimator(ntrees=1, mtries=len(x))
        self.model.train(x=x, y=y, training_frame=training_frame)

    def predict(self, frame):
        return self.model.predict(frame)

    def varimp(self):
        return self.model.varimp()


def evaluate_h2o_model(
        data,
        feature_names,
        target_col,
        model,
        n_iters=10,
        metric=roc_auc_score):
    """
    Train an H2O model on different train-test splits, and returns a metric
    evaluated on each fold, and the feature importance scores for each fold.

    Parameters
    ----------
    data : pd.DataFrame

    feature_names : list of strings
        Names of columns of dataframe that will make up X

    target_col : string
        Name of target column

    model : H2O model
        E.g. H2ORandomForestEstimator or H2ODecisionTree

    n_iters : int, default 10

    metric : function, default roc_auc_score
        A function that returns a float when called with metric(y_true, y_test)

    Returns
    -------
    metrics : list of floats

    feature_importances : list of dicts
        Each dict has the form {feature_name (str): feature_importance (float)}
    """
    h2ofr = h2o.H2OFrame(data)
    h2ofr.col_names = list(data.columns)
    metrics, feature_importances = list(), list()
    folds = StratifiedShuffleSplit(y=data[target_col],
                                   n_iter=n_iters,
                                   test_size=0.3)
    for train_idx, test_idx in folds:
        train_idx, test_idx = \
            sorted(train_idx), sorted(test_idx)  # H2O indices must be sorted
        model.train(x=feature_names,
                    y=target_col,
                    training_frame=h2ofr[train_idx, :])
        # Slicing an H2O frame causes a (depreciation) warning in h2o version
        # 3.10.0.3. There is a TODO to fix it, so we can probably safely ignore
        # the warning. The warning uses a print statement, so we'll temporarily
        # redirect stdout:
        with open(os.devnull, 'w') as f:
            old_out = sys.stdout
            sys.stdout = f
            predictions = model.predict(
                    h2ofr[test_idx, feature_names]).as_data_frame()
            sys.stdout = old_out
        try:
            prediction_scores = predictions['True']
        except KeyError:
            # Decision Trees only give a single 'predict' column
            prediction_scores = predictions['predict']
        metrics.append(
            metric(data[target_col].values[test_idx], prediction_scores))
        feature_importances.append(
            dict([(v[0], v[3]) for v in model.varimp()]))
    return {'metric': metrics, 'importances': feature_importances}


def evaluate_sklearn_model(
        data,
        feature_names,
        target_col,
        model,
        n_iters=10,
        metric=roc_auc_score):
    """
    Train an sklearn model on different train-test splits, and returns a metric
    evaluated on each fold, and the feature importance scores for each fold.

    Parameters
    ----------
    data : pd.DataFrame

    feature_names : list of strings
        Names of columns of dataframe that will make up X

    target_col : string
        Name of target column

    model : H2O model
        E.g. H2ORandomForestEstimator or H2ODecisionTree

    n_iters : int, default 10

    metric : function, default roc_auc_score
        A function that returns a float when called with metric(y_true, y_test)

    Returns
    -------
    results
        results.metrics : list of floats
        results.importances : list of dicts
            Each dict has the form
            {feature_name (str): feature_importance (float)}
    """
    metrics, feature_importances = list(), list()
    X = data[feature_names].values
    y = data[target_col].values
    folds = StratifiedShuffleSplit(y=y, n_iter=n_iters, test_size=0.3)
    for train_idx, test_idx in folds:
        model.fit(X[train_idx], y[train_idx])
        predictions = model.predict_proba(X[test_idx])
        metrics.append(
            metric(y[test_idx], predictions[:, 1]))
        try:
            feature_importances.append(
                dict(zip(feature_names, model.feature_importances_))
            )
        except AttributeError:  # Not a random forest!
            feature_importances.append(
                dict(zip(feature_names, model.coef_.ravel()))
            )
    return {'metric': metrics, 'importances': feature_importances}


def print_auc_mean_std(results):
    """Print an AUC-based summary of performance.

    Parameters
    ---------
    results : dict
       As produced by `evaluate_sklearn_model` or
       `evaluate_h2o_model`.

    Prints
    ------
    To standard output, a summary.
    """
    print("AUC: mean {:4.4f}, sd {:4.4f}".format(
        np.mean(results['metric']), np.std(results['metric'])))


def print_sorted_mean_importances(results, n=5):
    """Print a sorted list of features and their importance
    values (which might be coefficients, if this is a
    regression-type mode.

    Parameters
    ---------
    results : dict
       As produced by `evaluate_sklearn_model` or
       `evaluate_h2o_model`.

    n : int
       Number of results to print

    Prints
    ------
    To standard output, a formatted list.
    """
    data = defaultdict(list)
    imps = results['importances']
    for d in imps:
        for fname, imp in d.items():
            data[fname].append(imp)
    mu = {}
    for fname, vals in data.items():
        mu[fname] = np.mean(vals)
    mu = sorted(mu.items(), key=itemgetter(1), reverse=True)
    if n:
        mu = mu[: n]
    for fname, val in mu:
        print("{:>20}: {:0.03f}".format(fname, val))
