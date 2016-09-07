"""
This module replaces sklearn's average_precision_score and reflects code
submitted in this PR:
[insert URL]

See http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
and [insert URL] for more.
"""

import warnings

import numpy as np
from sklearn.metrics.base import _average_binary_score
from sklearn.metrics import precision_recall_curve


def auc(x, y, reorder=False, interpolation='linear',
        interpolation_direction='right'):
    """Estimate Area Under the Curve (AUC) using finitely many points and an
    interpolation strategy.

    This is a general function, given points on a curve.  For computing the
    area under the ROC-curve, see :func:`roc_auc_score`.

    Parameters
    ----------
    x : array, shape = [n]
        x coordinates.

    y : array, shape = [n]
        y coordinates.

    reorder : boolean, optional (default=False)
        If True, assume that the curve is ascending in the case of ties, as for
        an ROC curve. If the curve is non-ascending, the result will be wrong.

    interpolation : string ['trapezoid' (default), 'step']
        This determines the type of interpolation performed on the data.

        ``'linear'``:
            Use the trapezoidal rule (linearly interpolating between points).
        ``'step'``:
            Use a step function where we ascend/descend from each point to the
            y-value of the subsequent point.

    interpolation_direction : string ['right' (default), 'left']
        This determines the direction to interpolate from. The value is ignored
        unless interpolation is 'step'.

        ``'right'``:
            Intermediate points inherit their y-value from the subsequent point.
        ``'left'``:
            Intermediate points inherit their y-value from the previous point.

    Returns
    -------
    auc : float

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> pred = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    >>> metrics.auc(fpr, tpr)
    0.75

    See also
    --------
    roc_auc_score : Computes the area under the ROC curve

    precision_recall_curve :
        Compute precision-recall pairs for different probability thresholds

    """

    direction = 1
    if reorder:
        # reorder the data points according to the x axis and using y to
        # break ties
        order = np.lexsort((y, x))
        x, y = x[order], y[order]
    else:
        dx = np.diff(x)
        if np.any(dx < 0):
            if np.all(dx <= 0):
                direction = -1
            else:
                raise ValueError("Reordering is not turned on, and "
                                 "the x array is not increasing: %s" % x)

    if interpolation == 'linear':

        area = direction * np.trapz(y, x)

    elif interpolation == 'step':

        # we need the data to start in ascending order
        if direction == -1:
            x, y = list(reversed(x)), list(reversed(y))

        if interpolation_direction == 'right':
            # The left-most y-value is not used
            area = sum(np.diff(x) * np.array(y)[1:])

        elif interpolation_direction == 'left':
            # The right-most y-value is not used
            area = sum(np.diff(x) * np.array(y)[:-1])

        else:
            raise ValueError("interpolation_direction '{}' not recognised."
                             " Should be one of ['right', 'left']".format(
                                 interpolation_direction))

    else:
        raise ValueError("interpolation value '{}' not recognized. "
                         "Should be one of ['linear', 'step']".format(
                             interpolation))

    return area


def average_precision_score(y_true, y_score, average="macro",
                            sample_weight=None, interpolation="linear"):
    """Compute average precision (AP) from prediction scores

    This score corresponds to the area under the precision-recall curve, where
    points are joined using either linear or step-wise interpolation.

    Note: this implementation is restricted to the binary classification task
    or multilabel classification task.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : array, shape = [n_samples] or [n_samples, n_classes]
        True binary labels in binary label indicators.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or binary decisions.

    average : string, [None, 'micro', 'macro' (default), 'samples', 'weighted']
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    interpolation : string ['linear' (default), 'step']
        Determines the kind of interpolation used when computed AUC. If there are
        many repeated scores, 'step' is recommended to avoid under- or over-
        estimating the AUC. See www.roamanalytics.com/etc for details.

        ``'linear'``:
            Linearly interpolates between operating points.
        ``'step'``:
            Uses a step function to interpolate between operating points.

    Returns
    -------
    average_precision : float

    References
    ----------
    .. [1] `Wikipedia entry for the Average precision
           <http://en.wikipedia.org/wiki/Average_precision>`_

    See also
    --------
    roc_auc_score : Area under the ROC curve

    precision_recall_curve :
        Compute precision-recall pairs for different probability thresholds

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import average_precision_score
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> average_precision_score(y_true, y_scores)  # doctest: +ELLIPSIS
    0.79...

    """
    def _binary_average_precision(y_true, y_score, sample_weight=None):
        precision, recall, thresholds = precision_recall_curve(
            y_true, y_score, sample_weight=sample_weight)
        return auc(recall, precision, interpolation=interpolation,
                   interpolation_direction='right')

    if interpolation == "linear":
        # Check for number of unique predictions. If this is substantially less
        # than the number of predictions, linear interpolation is likely to be
        # biased.
        n_discrete_predictions = len(np.unique(y_score))
        if n_discrete_predictions < 0.75 * len(y_score):
            warnings.warn("Number of unique scores is less than 75% of the "
                          "number of scores provided. Linear interpolation "
                          "is likely to be biased in this case. You may wish "
                          "to use step interpolation instead. See docstring "
                          "for details.")
    return _average_binary_score(_binary_average_precision, y_true, y_score,
                                 average, sample_weight=sample_weight)
