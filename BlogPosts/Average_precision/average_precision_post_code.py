from copy import copy
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import display, HTML
matplotlib.style.use('../../src/roam.mplstyle')


def generate_data_and_constant_predictions(n, frac_positive):
    """
    Generates data in a fixed positive:negative ratio, and returns the
    data and scores from a dummy model that predicts 0.5 for all examples.

    Parameters
    ----------
    n : int
        Number of examples

    frac_positive : float
        Fraction of the examples that are positive

    Returns
    -------
    observations : list
        Consisting of (frac_positive * n) 1s, and (n - (frac_positive * n)) 0s

    constant_predictions : list
        Same length as observations
    """
    n_positive = int(frac_positive * n)
    n_negative = n - n_positive
    observations = [1 for _ in range(n_positive)] + \
                   [0 for _ in range(n_negative)]
    constant_predictions = [0.5 for _ in range(n_positive + n_negative)]
    return observations, constant_predictions


def plot_recall_precision_from_predictions(true, scores, **kwargs):
    """
    Computes precision and recall from some observations and scores assigned
    to them, and plots a precision-recall curve.

    Parameters
    ----------
    true : list
        Must be binary (i.e. 1s and 0s).

    scores : list
        Consisting of floats.

    kwargs : optional
        See plot_axes.

    """
    p, r, thresholds = precision_recall_curve(true, scores)
    plot_recall_precision(p, r, **kwargs)


def plot_recall_precision(p, r, **kwargs):
    """
    Plots a precision-recall graph from a series of operating points.

    Parameters
    ----------
    p : list
        Precision.

    r : recall
        Recall.

    kwargs : optional
        See plot_axes.

    Returns
    -------

    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    plot_axes(ax, p, r, legend_text='IAP', **kwargs)
    plt.show()


def plot_axes(
        ax, y, x,
        interpolation=None,
        marker_size=30,
        title=None,
        legend_text='Area'):
    """
    Plots a graph on axes provided.

    Parameters
    ----------
    ax : matplotlib axes

    y : list

    x : list

    interpolation : None (default) or string ['linear', 'step']

    marker_size : float (default: 30)

    title : None or string

    legend_text : string (default: 'Area')
        Text to include on the legend before showing the area. Only used
        if interpolation is not None.
    """
    ax.scatter(x, y, marker='o', linewidths=0, s=marker_size, clip_on=False)
    # Show first and last points more visably
    ax.scatter([x[i] for i in [0, -1]], [y[i] for i in [0, -1]],
               marker='x', linewidths=2, s=100, clip_on=False)
    ax.set_xlim((-0.05, 1.05))
    ax.set_ylim((-0.08, 1.08))
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    if title is not None:
        ax.set_title(title, fontsize=20)
    if interpolation is not None:
        if interpolation == 'linear':
            ax.plot(x, y)
            area = auc(x, y)
            ax.fill_between(x, 0, y, alpha=0.2,
                            label='{} = {:5.4f}'.format(legend_text, area))
            leg = ax.legend()
            leg.get_frame().set_linewidth(0.0)
        elif interpolation == 'step':
            p_long = [v for v in y for _ in (0, 1)][:-1]
            r_long = [v for v in x for _ in (0, 1)][1:]
            ax.plot(r_long, p_long)
            area = auc_using_step(x, y)
            ax.fill_between(r_long, 0, p_long, alpha=0.2, 
                            label='{} = {:5.4f}'.format(legend_text, area))
            leg = ax.legend()
            leg.get_frame().set_linewidth(0.0)
        else:
            print("Interpolation value of '{}' not recognised. "
                  "Choose from 'linear', 'quadrature'.".format(interpolation))


def compare_recall_precisions_from_predictions(true, score_dict, **kwargs):
    """
    Show two graphs side-by-side for two different sets of scores, against the
    same true observations.

    Parameters
    ----------
    true : list

    score_dict : dict
        Consisting of `{name: scores}` where `name` is a string and
        `scores` is a list of floats.

    kwargs : optional
        See plot_axes.
    """
    pr = OrderedDict()
    for name, score in score_dict.items():
        p, r, threshold = precision_recall_curve(true, score)
        pr[name] = [p, r]
    compare_recall_precision_graph(pr, **kwargs)


def compare_recall_precision_graph(pr_dict, title=None, **kwargs):
    """

    Parameters
    ----------
    pr_dict : dict
        Consisting of `{name: pr}` where `name` is a string and
        `pr` is a tuple of precision and recall values.

    title : string

    kwargs : optional
        See plot_axes.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    for side, (name, [p, r]) in enumerate(pr_dict.items()):
        plot_axes(ax[side], p, r, title=name, legend_text='IAP', **kwargs)
    if title is not None:
        fig.suptitle(title, fontsize=20, y=1.05)
    plt.show()


def operating_points(ranking):
    """
    Computes lists of precision and recall from an ordered list of observations.

    Parameters
    ----------
    ranking : list
        Entries should be binary (0 or 1) and in descending order
        (i.e. top-ranked is first).

    Returns
    -------
    precision : list

    recall : list
    """
    precision, recall = list(), list()
    for pos in range(len(ranking)):
        p, r = precision_recall_from_ranking(ranking, pos)
        precision.append(p)
        recall.append(r)
    return precision, recall


def precision_recall_from_ranking(ranking, position):
    """
    Computes the precision and recall of a particular assignment of labelled
    observations to a positive and negative class, where the positive class
    comes first in the list, and the negative class comes second, and the
    split point is specified.

    Parameters
    ----------
    ranking : list
        Ordered list of binary observations.

    position : int
        Position to split the list into positive and negative.

    Returns
    -------
    precision : float

    recall : float
    """
    if position == 0:
        precision = 1.0
        recall = 0.0
    else:
        ranking = np.array(ranking)
        precision = (ranking[:position] == 1).sum() / position
        recall = (ranking[:position] == 1).sum() / (ranking == 1).sum()
    return precision, recall


def auc_using_step(recall, precision):
    return sum([(recall[i] - recall[i+1]) * precision[i]
                for i in range(len(recall) - 1)])


def roam_average_precision(y_true, y_score, sample_weight=None):
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_score, sample_weight=sample_weight)
    return auc_using_step(recall, precision)


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


def subsample(X, y, frac_positive):
    """
    Subsamples a feature matrix and target vector to ensure that a specified
    fraction of the target values are positive.

    Parameters
    ----------
    X : np.array (n, m)

    y : np.array (n, )

    frac_positive : float

    Returns
    -------
    X : np.array (n', m)
        Some subset of the rows of the input X (i.e. n' <= n)

    y : np.array (n', )
        Some subset of the rows of the input y (i.e. n' <= n)
    """
    positive_idx = np.arange(len(y))[y == 1]
    negative_idx = np.arange(len(y))[y == 0]
    num_positive = int(frac_positive * len(negative_idx))
    positive_idx = np.random.choice(positive_idx, size=num_positive, replace=False)
    indices_to_use = np.concatenate([positive_idx, negative_idx])
    np.random.shuffle(indices_to_use)
    return X[indices_to_use], y[indices_to_use]


def generate_continuous_data_and_targets(
        n_dim,
        n_samples,
        mixing_factor=0.025,
        frac_positive=0.1):
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
    X, y = subsample(X, y, frac_positive)
    return X, y


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


def train_model_and_evaluate(n_dim=50, n_samples=10000, frac_positive=0.05,
                             mixing_factor=0.025):
    """
    Generates some data and trains a logistic regression model.

    Parameters
    ----------
    n_dim : int
        Number of dimensions for the training data.

    n_samples : int
        Number of observations.

    frac_positive : float

    mixing_factor : float
        Numbers nearer to 0 make the task more challenging.

    Returns
    -------
    y : np.array (n_test, )
        True observed values in the test set.

    y_scores : np.array (n_test, )
        Model predictions of the test samples.

    roc_auc : float
        ROC AUC score on the test data
    """
    X, y = generate_continuous_data_and_targets(
            n_dim=n_dim, n_samples=n_samples, frac_positive=frac_positive,
            mixing_factor=mixing_factor)
    splits = StratifiedShuffleSplit(y, test_size=0.3, random_state=42)
    train_idx, test_idx = list(splits)[0]
    lr = LogisticRegressionCV()
    lr.fit(X[train_idx], y[train_idx])
    y_scores = lr.predict_proba(X[test_idx])[:, 1]
    roc_auc = roc_auc_score(y[test_idx], y_scores)
    return y[test_idx], y_scores, roc_auc
