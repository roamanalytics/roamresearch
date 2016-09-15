"""
Companion code for the Roam blog post on hyperparameter optimization.
"""
import itertools as it
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import pandas as pd
import random
from scipy import stats
from time import time

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from skopt import gp_minimize, forest_minimize, gbrt_minimize
from skopt.space import Categorical
from sklearn.datasets import make_classification
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Ignore these warnings, which are related to the bleeding
# edge sklearn we're using.
import warnings
import sklearn.metrics.base
warnings.filterwarnings("ignore", category=DeprecationWarning)


__author__ = 'Ben Bernstein and Chris Potts'
__version__ = '1.0'
__license__ = 'Apache License Version 2.0, January 2004 http://www.apache.org/licenses/'


plt.style.use('../../src/roam.mplstyle')

LOG_LOSS = 'log_loss' # Newer versions will use 'neg_log_loss'


def artificial_dataset(
        n_samples=1000,
        n_features=100,
        n_classes=3,
        random_state=None):
    """
    sklearn random classification dataset generation using
    `sklearn.datasets.make_classification`.

    Parameters
    ----------
    n_samples : int
    n_features : int
    n_classes : int
    random_state : int or None

    Returns
    -------
    (X, y)
        The design matrix `X` and target `y`.
    """
    n_informative = int(0.8 * n_features)
    n_redundant = int(0.05 * n_features)    
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=n_informative, 
        n_redundant=n_redundant, 
        n_classes=n_classes, 
        random_state=random_state)
    return (X, y)


def assess(
        X, y,
        search_func,
        model_class,
        param_grid,
        xval_indices,
        loss=LOG_LOSS,
        test_metric=accuracy_score,
        dataset_name=None,
        search_func_args={}):
    """
    The core of the experimental framework. This runs cross-validation
    and, for the inner loop, does cross-validation to find the optimal
    hyperparameters according to `search_func`. These optimal
    parameters are then used for an assessment in the outer
    cross-validation run.

    Parameters
    ----------
    X : np.array
        The design matrix, dimension `(n_samples, n_features)`.

    y : list or np.array
        The target, of dimension `n_samples`.

    search_func : function
        The search function to use. Can be `grid_search`,
        `randomized_search`, `hyperopt_search`, `skopt_gp_minimize`,
        `skopt_forest_minimize`, or `skopt_forest_gbrt`, all
        defined in this module. This choice has to be compatible with 
        `param_grid`, in the sense that `grid_search` and
        `randomized_search` require a dict from strings to lists of
        values, `hyperopt_search` requires a dict from strings to
        hyperopt sampling functions, and the `skopt` functions
        require dicts from strings to (upper, lower) pairs of
        special `skopt` functions.
        
    model_class : classifier
        A classifier model in the mode of `sklearn`, with at least
        `fit` and `predict` methods operating on things like
        `X` and `y`.

    param_grid : dict
        Map from parameter names to appropriate specifications of
        appropriate values for that parameter. This is not the
        expanded grid, but rather the simple map that can be expanded
        by `expand_grid` below (though not all methods call for that).
        This has to be compatible with  `search_func`, and all the
        values must be suitable arguments to `model_class` instances.

    loss : function or string
        An appropriate loss function or string recognizable by
        `sklearn.cross_validation.cross_val_score`. In `sklearn`, scores
        are positive and losses are negative because they maximize,
        but here we are minimizing so we always want smaller to mean
        better.

    test_metric : function
        An `sklearn.metrics` function.

    xval_indices : list
        List of train and test indices into `X` and `y`. This defines
        the cross-validation. This is done outside of this method to
        allow for identical splits across different experiments.

    dataset_name : str or None
        Name for the dataset being analyzed. For book-keeping and
        display only.

    search_func_args : dict
        Keyword arguments to feed to `search_func`.       

    Returns
    -------
    dict
        Accumulated information about the experiment:

        {'Test accuracy': list of float,
         'Cross-validation time (in secs.)':list of float,
         'Parameters sampled': list of int,
         'Method': search_func.__name__,
         'Model': model_class.__name__,
         'Dataset': dataset_name,
         'Best parameters': list of dict,
         'Mean test accuracy': float,
         'Mean cross-validation time (in secs.)': float,
         'Mean parameters sampled': float}                   
    """
    data = {'Test accuracy': [],
            'Cross-validation time (in secs.)': [],
            'Parameters sampled': [],
            'Best parameters': [],
            'Method': search_func.__name__,
            'Model': model_class.__name__,
            'Dataset': dataset_name,
            'Best parameters':[]}
    for cv_index, (train_index, test_index) in enumerate(xval_indices, start=1):
        print("\t{}".format(cv_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        start = time()                
        results = search_func(
            X_train,
            y_train,
            model_class,
            param_grid,
            loss,
            **search_func_args)    
        data['Cross-validation time (in secs.)'].append(time() - start)
        data['Parameters sampled'].append(len(results))
        best_params = sorted(results, key=itemgetter('loss'), reverse=False)
        best_params = best_params[0]['params']
        data['Best parameters'].append(best_params)
        bestmod = model_class(**best_params)
        bestmod.fit(X_train, y_train)
        predictions = bestmod.predict(X_test)        
        data['Test accuracy'].append(test_metric(y_test, predictions))
    data['Mean test accuracy'] = np.mean(
        data['Test accuracy'])
    data['Mean cross-validation time (in secs.)'] = np.mean(
        data['Cross-validation time (in secs.)'])
    data['Mean parameters sampled'] = np.mean(
        data['Parameters sampled'])
    return data


def get_cross_validation_indices(X, y, n_folds=5, random_state=None):
    """
    Use `StratifiedKFold` to create an `n_folds` cross-validator for
    the dataset defined by `X` and y`. Only `y` is used, but both are
    given for an intuitive interface; `X` could just as easily be used.
    """
    return StratifiedKFold(y, n_folds=n_folds, random_state=random_state)


def random_search(
        X_train, y_train, model_class, param_grid, loss, sampsize=None):
    """
    Random search over the grid defined by `param_grid`.

    Parameters
    ----------
    X_train : np.array
        The design matrix, dimension `(n_samples, n_features)`.

    y_train : list or np.array
        The target, of dimension `n_samples`.

    model_class : classifier
        A classifier model in the mode of `sklearn`, with at least
        `fit` and `predict` methods operating on things like
        `X` and `y`.

    param_grid : dict
        Map from parameter names to lists of appropriate values
        for that parameter. This is not the expanded grid, but
        rather the simple map that can be expanded by `expand_grid`
        below. This method performs the expansion.

    loss : function or string
        An appropriate loss function or string recognizable by
        sklearn.cross_validation.cross_val_score. In sklearn, scores
        are positive and losses are negative because they maximize,
        but here we are minimizing so we always want smaller to mean
        better.

    sampsize : int or None
        Number of samples to take from the grid. If `None`, then
        `sampsize` is half the size of the full grid.

    Returns
    -------
    list of dict
        Each has keys 'loss' and 'params', where 'params' stores the
        values from `param_grid` for that run. The primary organizing
        value is 'loss'.

    Example
    -------
    >>> param_grid = {
            'max_depth' : [4, 8],
            'learning_rate' : [0.01, 0.3],
            'n_estimators' : [20, 50],
            'objective' : ['multi:softprob'],
            'gamma' : [0, 0.25],
            'min_child_weight' : [1],
            'subsample' : [1],
            'colsample_bytree' : [1]}
    >>> res = random_search(X, y, XGBClassifier, param_grid, LOG_LOSS)

    To be followed by (see below):

    >>> best_params, best_loss = best_results(res)
    """
    exapnded_param_grid = expand_grid(param_grid)
    if sampsize == None:
        sampsize = int(len(exapnded_param_grid) / 2.0)
    samp = random.sample(exapnded_param_grid, sampsize)
    results = []
    for params in samp:
        err = cross_validated_scorer(
            X_train, y_train, model_class, params, loss)
        results.append({'loss': err, 'params': params})
    return results


def grid_search(X_train, y_train, model_class, param_grid, loss):
    """
    Full grid search over the grid defined by `param_grid`.

    Parameters
    ----------
    X_train : np.array
        The design matrix, dimension `(n_samples, n_features)`.

    y_train : list or np.array
        The target, of dimension `n_samples`.

    model_class : classifier
        A classifier model in the mode of `sklearn`, with at least
        `fit` and `predict` methods operating on things like
        `X` and `y`.

    param_grid : dict
        Map from parameter names to lists of appropriate values
        for that parameter. This is not the expanded grid, but
        rather the simple map that can be expanded by `expand_grid`
        below. This method performs the expansion.

    loss : function or string
        An appropriate loss function or string recognizable by
        sklearn.cross_validation.cross_val_score. In sklearn, scores
        are positive and losses are negative because they maximize,
        but here we are minimizing so we always want smaller to mean
        better.

    Returns
    -------
    list of dict
        Each has keys 'loss' and 'params', where 'params' stores the
        values from `param_grid` for that run. The primary organizing
        value is 'loss'.

    Example
    -------
    >>> param_grid = {
            'max_depth' : [4, 8],
            'learning_rate' : [0.01, 0.3],
            'n_estimators' : [20, 50],
            'objective' : ['multi:softprob'],
            'gamma' : [0, 0.25],
            'min_child_weight' : [1],
            'subsample' : [1],
            'colsample_bytree' : [1]}
    >>> res = grid_search(X, y, XGBClassifier, param_grid, LOG_LOSS)

    To be followed by (see below):

    >>> best_params, best_loss = best_results(res)
    """
    results = []
    expanded_param_grid = expand_grid(param_grid)
    for params in expanded_param_grid:
        err = cross_validated_scorer(
            X_train, y_train, model_class, params, loss)
        results.append({'loss': err, 'params': params})
    return results


def expand_grid(param_grid):
    """
    Expand `param_grid` to the full grid, as a list of dicts.

    Parameters
    ----------
    param_grid : dict
        Map from parameter names to lists of appropriate values
        for that parameter. This is not the expanded grid, but
        rather the simple map that can be expanded by `expand_grid`
        below. This method performs the expansion.

    Returns
    -------
    list of dict
        If `param_grid` was

        {'foo': [1,2], 'bar': [3,4]}

        Then the return value would be

        [{'foo': 1, 'bar': 3},  {'foo': 1, 'bar': 4},
         {'foo': 2, 'bar': 3},  {'foo': 2, 'bar': 4}]
    """        
    varNames = sorted(param_grid)
    return [dict(zip(varNames, prod))
            for prod in it.product(*(param_grid[varName]
                                     for varName in varNames))]


def cross_validated_scorer(
        X_train, y_train, model_class, params, loss, kfolds=5):
    """
    The scoring function used through this module, by all search
    functions.

    Parameters
    ----------
    X_train : np.array
        The design matrix, dimension `(n_samples, n_features)`.

    y_train : list or np.array
        The target, of dimension `n_samples`.

    model_class : classifier
        A classifier model in the mode of `sklearn`, with at least
        `fit` and `predict` methods operating on things like
        `X` and `y`.

    params : dict
        Map from parameter names to single appropriate values
        for that parameter. This will be used to build a model
        from `model_class`.

    loss : function or string
        An appropriate loss function or string recognizable by
        sklearn.cross_validation.cross_val_score. In sklearn, scores
        are positive and losses are negative because they maximize,
        but here we are minimizing so we always want smaller to mean
        better.

    kfolds : int
        Number of cross-validation runs to do.

    Returns
    -------
    float
       Average loss over the `kfolds` runs.
    """
    mod = model_class(**params)
    cv_score = -1 * cross_val_score(
        mod,
        X_train,
        y=y_train,
        scoring=loss,
        cv=kfolds,
        n_jobs=1).mean()
    return cv_score


def hyperopt_search(
        X_train, y_train, model_class, param_grid, loss, max_evals=100):
    """
    Search according to hyperopt's Tree of Parzen Estimator.

    Parameters
    ----------
    X_train : np.array
        The design matrix, dimension `(n_samples, n_features)`.

    y_train : list or np.array
        The target, of dimension `n_samples`.

    model_class : classifier
        A classifier model in the mode of `sklearn`, with at least
        `fit` and `predict` methods operating on things like
        `X` and `y`.

    param_grid : dict
        Map from parameter names to `hyperopt` sampling functions.
        The parameter names need to work with `model_class`, and the
        values specifying how to sample values.

    loss : function or string
        An appropriate loss function or string recognizable by
        sklearn.cross_validation.cross_val_score. In sklearn, scores
        are positive and losses are negative because they maximize,
        but here we are minimizing so we always want smaller to mean
        better.

    max_evals : int
        Number of evaluations to do.

    Returns
    -------
    list of dict
        Each has keys 'loss' and 'params', where 'params' stores the
        values from `param_grid` for that run. The primary organizing
        value is 'loss'. These values are accumulated and stored by
        the `Trials` instance used in the call to `fmin`. A 'status'
        record is also retained but not used elsewhere in the module.

    Example
    -------
    >>> hyperopt_grid = {
            'max_depth' : hp.choice('max_depth', range(4, 13, 1)),
            'learning_rate' : hp.quniform('learning_rate', 0.01, 0.5, 0.01),
            'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
            'objective' : 'multi:softprob',
            'gamma' : hp.quniform('gamma', 0, 0.50, 0.01),
            'min_child_weight' : hp.quniform('min_child_weight', 1, 5, 1),
            'subsample' : hp.quniform('subsample', 0.1, 1, 0.01),
            'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)}
    >>> res = hyperopt_search(X, y, XGBClassifier, hyperopt_grid, LOG_LOSS, max_evals=10)

    To be followed by (see below):

    >>> best_params, best_loss = best_results(res)
    """
    def objective(params):
        err = cross_validated_scorer(
            X_train, y_train, model_class, params, loss)
        return {'loss': err, 'params': params, 'status': STATUS_OK}
    trials = Trials()
    results = fmin(
        objective, param_grid, algo=tpe.suggest,
        trials=trials, max_evals=max_evals)
    return trials.results


def skopt_search(
        X_train, y_train, model_class, param_grid, loss, skopt_method, n_calls=100):
    """
    General method for applying `skopt_method` to the data.

    Parameters
    ----------
    X_train : np.array
        The design matrix, dimension `(n_samples, n_features)`.

    y_train : list or np.array
        The target, of dimension `n_samples`.

    model_class : classifier
        A classifier model in the mode of `sklearn`, with at least
        `fit` and `predict` methods operating on things like
        `X` and `y`.

    param_grid : dict
        Map from parameter names to pairs of values specifying the
        upper and lower ends of the space from which to sample.
        The values can also be directly specified as `skopt`
        objects like `Categorical`.

    loss : function or string
        An appropriate loss function or string recognizable by
        sklearn.cross_validation.cross_val_score. In sklearn, scores
        are positive and losses are negative because they maximize,
        but here we are minimizing so we always want smaller to mean
        better.

    skopt_method : skopt function
        Can be `gp_minimize`, `forest_minimize`, or `gbrt_minimize`.

    n_calls : int
        Number of evaluations to do.

    Returns
    -------
    list of dict
        Each has keys 'loss' and 'params', where 'params' stores the
        values from `param_grid` for that run. The primary organizing
        value is 'loss'.
    """
    param_keys, param_vecs = zip(*param_grid.items())
    param_keys = list(param_keys)
    param_vecs = list(param_vecs)

    def skopt_scorer(param_vec):
        params = dict(zip(param_keys, param_vec))
        err = cross_validated_scorer(
            X_train, y_train, model_class, params, loss)
        return err
    outcome = skopt_method(skopt_scorer, list(param_vecs), n_calls=n_calls)
    results = []
    for err, param_vec in zip(outcome.func_vals, outcome.x_iters):
        params = dict(zip(param_keys, param_vec))
        results.append({'loss': err, 'params': params})
    return results


def skopt_gp_search(
        X_train, y_train, model_class, param_grid, loss, n_calls=100):
    """`skopt` according to the Gaussian Process search method. For
    details on the parameters, see `skopt_search`.

    Example
    -------
    >>> skopt_grid = {
            'max_depth': (4, 12),
            'learning_rate': (0.01, 0.5),
            'n_estimators': (20, 200),
            'objective' : Categorical(('multi:softprob',)),
            'gamma': (0, 0.5),
            'min_child_weight': (1, 5),
            'subsample': (0.1, 1),
            'colsample_bytree': (0.1, 1)}
    >>> res = skopt_gp_search(X, y, XGBClassifier, skopt_grid, LOG_LOSS, n_calls=10)

    To be followed by (see below):

    >>> best_params, best_loss = best_results(res)
    """
    return skopt_search(
        X_train, y_train, model_class, param_grid, loss, gp_minimize, n_calls=n_calls)


def skopt_forest_search(
        X_train, y_train, model_class, param_grid, loss, n_calls=100):
    """`skopt` according to the decision tree search method. For
    details on the parameters, see `skopt_search`.

    Example
    -------
    >>> skopt_grid = {
            'max_depth': (4, 12),
            'learning_rate': (0.01, 0.5),
            'n_estimators': (20, 200),
            'objective' : Categorical(('multi:softprob',)),
            'gamma': (0, 0.5),
            'min_child_weight': (1, 5),
            'subsample': (0.1, 1),
            'colsample_bytree': (0.1, 1)}
    >>> res = skopt_forest_search(X, y, XGBClassifier, skopt_grid, LOG_LOSS, n_calls=10)

    To be followed by (see below):

    >>> best_params, best_loss = best_results(res)
    """
    return skopt_search(
        X_train, y_train, model_class, param_grid, loss, forest_minimize, n_calls=n_calls)


def skopt_gbrt_search(
        X_train, y_train, model_class, param_grid, loss, n_calls=100):
    """`skopt` according to the gradient-boosting-tree search method.
    For details on the parameters, see `skopt_search`.

    Example
    -------
    >>> skopt_grid = {
            'max_depth': (4, 12),
            'learning_rate': (0.01, 0.5),
            'n_estimators': (20, 200),
            'objective' : Categorical(('multi:softprob',)),
            'gamma': (0, 0.5),
            'min_child_weight': (1, 5),
            'subsample': (0.1, 1),
            'colsample_bytree': (0.1, 1)}
    >>> res = skopt_gbrt_search(X, y, XGBClassifier, skopt_grid, LOG_LOSS, n_calls=10)

    To be followed by (see below):

    >>> best_params, best_loss = best_results(res)
    """
    return skopt_search(
        X_train, y_train, model_class, param_grid, loss, gbrt_minimize, n_calls=n_calls)


def prepare_summary(results):
    """Format the `results` dictionary into a `pandas` `DataFrame`,
    with columns 'Method', 'Mean parameters sampled', 'Mean test accuracy',
    'Mean cross-validation time (in secs.)'."""
    results = {k:v for k,v in results.items()
               if k not in {'Test accuracy',
                            'Cross-validation time (in secs.)',
                            'Parameters sampled'}}
    df = pd.DataFrame([results])
    df = df[['Method',
             'Mean parameters sampled',
             'Mean test accuracy',
             'Mean cross-validation time (in secs.)']]
    return df


def prepare_summaries(results_list):
    """Format all the results dictionaries in `results_list` into a
    single pandas `DataFrame`.
    """
    dfs = []
    for results in results_list:
        dfs.append(prepare_summary(results))
    combo = pd.concat(dfs, axis=0)
    combo.index = range(1,len(combo)+1)
    return combo


def run_experiments(
        experimental_run,
        dataset,
        model_class=XGBClassifier,
        loss=LOG_LOSS,
        test_metric=accuracy_score,
        random_state=None,
        dataset_name=None):
    """
    Basic experimental framework.

    Parameters
    ----------
    experimental_run : list of tuples
        These tuples should have exactly three members: the first one
        of `grid_search`, `randomized_search`, `hyperopt_search`,
        `skopt_gp_minimize`, `skopt_forest_minimize`, or
        `skopt_forest_gbrt`, the second an appropriate `param_grid`
        dict for that function, and the third a dict specifying
        keyword arguments to the search function.

    dataset : (np.array, iterable)
        A dataset (X, y) where `X` has dimension
        `(n_samples, n_features)` and `y` has
         dimension `n_samples`.
    
    model_class : classifier
        A classifier model in the mode of `sklearn`, with at least
        `fit` and `predict` methods operating on things like
        `X` and `y`.

    loss : function or string
        An appropriate loss function or string recognizable by
        `sklearn.cross_validation.cross_val_score`. In `sklearn`, scores
        are positive and losses are negative because they maximize,
        but here we are minimizing so we always want smaller to mean
        better.

    test_metric : function
        An `sklearn.metrics` function.

    random_state : int

    dataset_name : str or None
        Informal name to give the dataset. Purely for
        book-keeping.

    Returns
    -------
    list of dict
       Each dict is a results dictionary of the sort returned
       by `assess`.
    """                    
    X, y = dataset    
    skf = get_cross_validation_indices(
        X, y, random_state=random_state)        
    all_results = []
    # This loop can easily be parallelized, but doing so can
    # be tricky on some systems, since `cross_val_score`
    # calls `joblib` even if `n_jobs=1`, resulting in
    # nested parallel jobs even if there is no actual
    # parallelization elsewhere in the experimental run.
    for search_func, param_grid, kwargs in experimental_run:
        print(search_func.__name__)
        all_results.append(
            assess(
                X, y,                
                search_func=search_func, 
                model_class=XGBClassifier, 
                param_grid=param_grid,
                xval_indices=skf,
                loss=loss,
                test_metric=test_metric,                
                dataset_name=dataset_name,
                search_func_args=kwargs))
    return all_results


def representation_size_experiments(
        experimental_run,
        n_samples=100, 
        min_features=50,
        max_features=100, 
        increment=50,
        loss=LOG_LOSS,
        test_metric=accuracy_score,
        random_state=None):
    """Run a series of experiments with `experimental_run`
    exploring `n_feature` sizes from `min_features` to
    `max_features` (inclusive) in increments of `increment`.

    Parameters
    ----------
    experimental_run : list of tuples
        These tuples should have exactly three members: the first one
        of `grid_search`, `randomized_search`, `hyperopt_search`,
        `skopt_gp_minimize`, `skopt_forest_minimize`, or
        `skopt_forest_gbrt`, the second an appropriate `param_grid`
        dict for that function, and the third a dict specifying
        keyword arguments to the search function.

    n_samples : int
        Number of examples.

    min_features : int
        Smallest feature representation size.

    max_features : int
        Largest feature representation size.
    
    increment : int
        Increments between `min_features` and `max_features`.

    loss : function or string
        An appropriate loss function or string recognizable by
        `sklearn.cross_validation.cross_val_score`. In `sklearn`, scores
        are positive and losses are negative because they maximize,
        but here we are minimizing so we always want smaller to mean
        better.

    test_metric : function
        An `sklearn.metrics` function.
        
    random_state : int

    Returns
    -------
    list of values returned by `run_experiments`.               
    """                
    all_results = []
    for n_features in range(min_features, max_features+1, increment):
        dataset = artificial_dataset(
            n_samples=100,
            n_features=n_features,
            n_classes=3,
            random_state=random_state)
        results = run_experiments(
            experimental_run,
            dataset,
            loss=loss,
            test_metric=accuracy_score,
            random_state=random_state,
            dataset_name=n_features)
        all_results.append(results)
    return all_results
        

def plot_representation_size_accuracy(
        representation_size_results, include_cis=True):
    """Plot the test accuracy values from the output of
    `representation_size_experiments`"""
    kwargs = {
        'metric_name': 'Test accuracy',
        'metric_mean_name': 'Mean test accuracy',
        'ylabel': 'Test accuracy',
        'value_transform': (lambda x : x),
        'include_cis': include_cis,
        'ylabel_overlap_threshold': 0.02}    
    plot_representation_size(representation_size_results, **kwargs)

    
def plot_representation_size_time(
        representation_size_results, include_cis=True):
    """Plot the cross-validation time values from the output of
    `representation_size_experiments`"""
    kwargs = {
        'metric_name': 'Cross-validation time (in secs.)',
        'metric_mean_name': 'Mean cross-validation time (in secs.)',
        'ylabel': 'Cross-validation time (log-scale)',
        'value_transform': (lambda x : np.log(x)),
        'include_cis': include_cis,
        'ylabel_overlap_threshold': 0.2}    
    plot_representation_size(representation_size_results, **kwargs)


def plot_representation_size(
        representation_size_results,
        metric_name='',
        metric_mean_name='',
        ylabel='',
        value_transform=(lambda x : x),
        include_cis=True,
        ylabel_overlap_threshold=0.2):
    """Generic interface for plotting the output of
    `representation_size_experiments`"""
    fig, ax = plt.subplots(1)
    fig.set_figwidth(10)
    fig.set_figheight(8)    
    methods = set([d['Method'] for results in representation_size_results
                   for d in results])
    colors = {
        'grid_search': '#212121',
        'random_search': '#876DB5',
        'hyperopt_search': '#4D8951',
        'skopt_forest_minimize': '#0499CC',
        'skopt_gp_minimize': '#03A9F4',
        'skopt_forest_gbrt': '#32A8B4'}
    text_positions = []  
    for method in methods:
        color = colors[method]
        method_results = [d for results in representation_size_results
                          for d in results if d['Method']==method]
        method_results = sorted(method_results, key=itemgetter('Dataset'))
        xpos = [d['Dataset'] for d in method_results]
        mus = [value_transform(d[metric_mean_name]) for d in method_results]                
        if include_cis:
            cis = [value_transform(get_ci(d[metric_name]))
                   for d in method_results]
            for x, ci in zip(xpos, cis):
                ax.plot([x,x], ci, color=color)
        ax.plot(
            xpos, mus, marker='.', linestyle='-', markersize=12, color=color)
        text_positions.append(mus[-1])
    method_text_positions = [[p,m] for p, m in zip(text_positions, methods)]
    method_text_positions = sorted(method_text_positions)
    for i in range(len(method_text_positions)-1):
        here = method_text_positions[i][0]
        there = method_text_positions[i+1][0]
        if there - here < ylabel_overlap_threshold:
            method_text_positions[i+1][0] = here + ylabel_overlap_threshold
    xpad = min(xpos)*0.2
    for text_pos, method in method_text_positions:
        ax.text(max(xpos)+(xpad*2), text_pos, method,
                fontsize=16, color=colors[method],
                va='center', weight='bold')
    ax.set_xlim([min(xpos)-xpad, max(xpos)+xpad])
    ax.set_xlabel("Number of features")
    ax.set_ylabel(ylabel)
    

def get_ci(vals, percent=0.95):
    """Confidence interval for `vals` from the Students' t
    distribution. Uses `stats.t.interval`.

    Parameters
    ----------
    percent : float
       Size of the confidence interval. The default is 0.95. The only
       requirement is that this be above 0 and at or below 1.

    Returns
    -------
    tuple
        The first member is the upper end of the interval, the second
        the lower end of the interval.
    
    """
    if len(set(vals)) == 1:
        return (vals[0], vals[0])
    mu = np.mean(vals)
    df = len(vals)-1
    sigma = np.std(vals) / np.sqrt(len(vals))
    return stats.t.interval(percent, df, loc=mu, scale=sigma)
        
    

