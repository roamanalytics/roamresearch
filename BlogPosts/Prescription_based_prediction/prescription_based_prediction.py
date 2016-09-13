"""
The is the experimental framework associated with the Roam blog post
'Prescription-based prediction': URL
"""
from collections import Counter
from colour import Color
import json
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import pandas as pd
from scipy import stats
import six
from statsmodels.graphics.mosaicplot import mosaic
import sys

from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score

from sklearn.feature_extraction.text import TfidfTransformer

from lime.lime_tabular import LimeTabularExplainer

import tensorflow as tf

# Ignore these warnings, which fill the screen and which are unavoidable
# given the problems we're addressing:
import warnings
import sklearn.metrics.base
warnings.filterwarnings(
    "ignore", category=sklearn.metrics.base.UndefinedMetricWarning)


__author__ = 'Nick Dingwall, Chris Potts, and Devini Senaratna'
__version__ = '1.0'
__license__ = 'Apache License Version 2.0, January 2004 http://www.apache.org/licenses/'


plt.style.use('../../src/roam.mplstyle')


dataset_filename = 'roam_prescription_based_prediction.jsonl'


def iter_dataset():
    """Iterate through the dataset, JSON line by JSON line, yielding
    tuples consisting of the features (dict mapping drug names to
    counts) and the target variables (dict mapping variable names to
    values)."""
    with open(dataset_filename, 'rt') as f:
        for line in f:
            ex = json.loads(line)
            yield (ex['cms_prescription_counts'],
                   ex['provider_variables'])


def build_experimental_dataset(drug_mincount=50, specialty_mincount=50):
    """Process the dataset so that its in a suitable format for
    supervised learning.

    Parameters
    ----------
    drug_mincount : int
        Keep only providers who prescribed this many different drugs.

    specialty_mincount : int
        Keep only providers whose specialty is one with at least
        this many occurrences in the dataset.

    Returns
    -------
    (X, ys, vectorizer)
        Where `X is a sparse array, examples by features, `ys` is a
        pandas DataFrame of target variables, with rows aligned to
        `X` and columns giving individual targets, and `vectorizer`
        is an `sklearn.feature_extraction.DictVectorizer`, retained
        for its feature names and its ability to featurize new data.
    """
    # Initial full dataset:
    data = [(phi_dict, y_dict) for phi_dict, y_dict in iter_dataset()
            if len(phi_dict) >= drug_mincount]
    # Frequency distribution of the specialties:
    specialties = Counter([y_dict['specialty'] for _, y_dict in data])
    # Limit to those with at least `specialty_mincount` examples:
    specialties = set([s for s,c in specialties.items()
                       if c >= specialty_mincount]) 
    data = [(phi, ys) for phi, ys in data
            if ys['specialty'] in specialties]
    # Process the dataset into an array and a pandas frame:
    feats, ys = zip(*data)
    vectorizer = DictVectorizer(sparse=True)    
    X = vectorizer.fit_transform(feats)
    X = TfidfTransformer().fit_transform(X)        
    ys = pd.DataFrame(list(ys))
    return (X, ys, vectorizer)


def cross_validate(
        X, y, mod, param_grid={}, randomsearch=False, n_iter=10,
        n_folds=5, n_jobs=1, random_state=None, verbose=True):
    """Evaluate `mod` on `X` and `y` via cross-validation, where
    the hyperparameters included in `param_grid` are optimized
    via cross-validated gridsearch on each training fold. The
    evaluation metric is macro F1, suitable for seeing how we
    do on diverse and unevenly distributed multi-class problems.
    
    Parameters
    ----------
    X : array-like (dense or sparse matrix)
        The design matrix.

    y : np.array
        The target variable, aligned with `X` row-wise.

    mod : supervised `sklearn` or `sklearn`-style model
        The restrictions: (A) this model must have `fit`, `predict`,
        `get_params`and `set_params` methods that behave like those in
        `sklearn`. This is so that the model can be fed to
        `sklearn.grid_search.GridSearchCV`. (B) The keys and values
        in `param_grid` must correspond to suitable parameters to
        this model.

    param_grid : dict
        A map from keys (parameter names for initializing `mod`) to
        values (lists of appropriate parameter values for `mod`).

    randomsearch : bool
        If True, then random search over the grid is used. Otherwise,
        the full grid of settings is explored.

    n_iter : int
        Number of hyperparameter settings to try. Ignored if
        `randomized=False`.

    n_folds : int
        Number of cross-validation folds for the primary experiment
        and for cross-validation of hyperparameters during training.

    n_jobs : int
        Number of parallel jobs to run during gridsearch.

    random_state : int
        Set this to create the same cross-validation splits for every
        run (useful for reproducibility).

    verbose : bool
        Whether to print summary information after each fold.

    Returns
    -------
    (np.array, list of dict)
        A list of macro F1 scores, and a list of trained models
        (the one chosen predictions in each fold).
    """      
    scores = np.zeros(n_folds)
    models = []
    skf = StratifiedKFold(y, n_folds=n_folds, random_state=random_state)
    for cv_index, (train_index, test_index) in enumerate(skf):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]        
        bestmod = mod
        # Hyper-parameter selection on the training set:
        if param_grid:
            crossvalidator = None
            if randomsearch:
                crossvalidator = RandomizedSearchCV(
                    mod, param_grid, n_iter=n_iter, cv=n_folds,
                    scoring='f1_macro', n_jobs=n_jobs)                
            else:
                crossvalidator = GridSearchCV(
                    mod, param_grid, cv=n_folds,
                    scoring='f1_macro', n_jobs=n_jobs)                
            crossvalidator.fit(X_train, y_train)
            bestmod = crossvalidator.best_estimator_        
        bestmod.fit(X_train, y_train)
        # Test predictions and scoring:
        predictions = bestmod.predict(X_test)
        score = f1_score(
            y_test, predictions, average='macro', pos_label=None)
        scores[cv_index] = score
        models.append(bestmod)
        # Per-fold summary:        
        if verbose:
            bestmod_summary = {k: bestmod.get_params()[k]
                               for k in param_grid}
            print("Fold {0:} score {1:0.03f}; best params: {2:}".format(
                (cv_index+1), score, bestmod_summary))
        # Avoid tensorflow warnings derived from old graphs:
        try:
            bestmod.sess.close()
        except AttributeError:
            pass
    return (scores, models)


def run_all_experiments(
        X, ys, mod,
        targets=('gender', 'region', 'specialty'), param_grid={},
        n_iter=10, randomsearch=False, n_folds=5, n_jobs=5,
        random_state=None, verbose=True):
    """Run cross-validated experiments on each variable in `ys`.

    Parameters
    ----------
    The parameters are the same as for `cross_validate`, with the
    exception of `ys`, which is a pandas DataFrame of the sort
    created by `build_experimental_dataset` and `targets` is a
    subset of the column names in `ys` to be included in the
    evaluation.

    Returns
    -------
    dict
        A list of dicts, each one summarizing the results of the
        experiment:
        {
            'model': `mod.__class__.__name__`,
            'y': <name of target variable>,
            'score1': float
            ...
            'scoreN': float, # where N=`nfolds`
            'score_mean': float            
        }
    """
    summary = []
    for target_name in targets:
        y = ys[target_name].values
        this_summary = {'model': mod.__class__.__name__,
                        'target': target_name,
                        'n_classes': len(set(y))}
        if verbose:
            print('='*70)
            print(target_name)
        scores, models = cross_validate(
            X, y, mod, param_grid=param_grid,
            randomsearch=randomsearch, n_iter=n_iter,
            n_folds=n_folds, n_jobs=n_jobs,
            random_state=random_state, verbose=verbose)
        for i, s in enumerate(scores, start=1):
            this_summary['score{}'.format(i)] = s
        this_summary['score_mean'] = scores.mean()
        this_summary['models'] = models
        summary.append(this_summary)
    return summary


def summarize_crossvalidation(scores):
    """Prints the mean and Student t 95% confidence interval
    for np.array `scores`.
    """
    return "Mean F1: {0:0.02f}; " \
      "95% confidence interval: {1:0.004f}-{2:0.004f}".format(
          scores.mean(), *get_ci(scores))


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


def prepare_summaries(summaries, include_cis=False):
    """Pool a bunch of summaries.

    Parameters
    ----------
    summary : tuple of dict
        A list of dicts as produced by `run_all_experiments`.

    include_cis : bool
        Whether to include confidence intervals.

    Returns
    -------
    pd.DataFrame
    """     
    dfs = []
    for i, summary in enumerate(summaries):
        # Target names only for the rightmost summary:
        include_target_names = True if i == 0 else False
        df = prepare_summary(
            summary,
            include_target_names=include_target_names,
            include_cis=include_cis)
        dfs.append(df)
    combo = pd.concat(dfs, axis=1)
    return combo
        

def prepare_summary(
        summary, include_target_names=False, include_cis=False):
    """Format the output of `run_all_experiments` as a DataFrame.

    Parameters
    ----------
    summary : dict
        As produced by `run_all_experiments`.

    include_target_names : bool
        Whether to include the target variable names in the
        rightmost column, along with the number of classes.

    include_cis : bool
        Whether to include confidence intervals.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame(summary)
    model = df[['model']].values[0][0]
    cols = ['score_mean']
    if include_cis:
        cis = [get_ci(row) for row in df.filter(regex='score\d+').values]  
        df['CI_lower'] = [i[0] for i in cis]
        df['CI_upper'] = [i[1] for i in cis]
        cols += ['CI_lower', 'CI_upper']
    if include_target_names:
        cols = ['target', 'n_classes'] + cols
    df = df[cols]
    cols[cols.index('score_mean')] = model
    df.columns = cols    
    return df


def inspect_regression_weights(models, vectorizer):
    """Average the learned coefficients for the regression models in
    `models` and return a mapping from classes to `DataFrame`s
    containing features lists sorted by the weights for that class.

    Parameters
    ----------
    models : list
        A list of sklearn models with `coef_` and `classes_`
        attributes. Intuitively, these are *trained* generalized
        linear models or SVMs, in the `sklearn` context.

    vectorizer : `DictVectorizer`
        Must have a `get_feature_names` method, and the resulting
        list must be aligned with the `coef_` methods for `models`.
        In general, this means that each of these models should
        have been trained on a matrix produced by `vectorizer`.

    Returns
    -------
    dict
      A map from class names to two-column `pd.DataFrame` instances,
      with the rows sorted by feature weight, largest at the top.
    """
    all_coefs = np.array([mod.coef_ for mod in models])
    mean_coefs = all_coefs.mean(axis=0)    
    fnames = vectorizer.get_feature_names()
    data = {}
    classes = models[0].classes_
    for i, cls in enumerate(classes):
        vals = sorted(zip(fnames, mean_coefs[i].ravel()), 
                      key=itemgetter(1), reverse=True)
        data[cls] = pd.DataFrame(vals, columns=[cls, 'Weight'])
    return data


def lime_inspection(
        X, ys, vectorizer, summary, target, label, num_features=5, top_labels=1):
    """Inspect a random example with label `label` using LIME.

    Parameters
    ----------
    X : array-like (dense or sparse matrix)
        The design matrix.

    ys : a pd.DataFrame
        As produced by `build_experimental_dataset`.

    vectorizer : `DictVectorizer`
        Must have a `get_feature_names` method, and the resulting
        list must be aligned with the `X`.

    summary : list of dict
        As produced by `run_all_experiments`.

    target : str
        A value of 'target' in one of the dicts in `summary`.

    label : str
        A value that `target can take on.

    num_features : int
        Number of features to display.

    top_labels : int
        Number of classes to display -- the top classes predicted
        by the model.

    Displays
    --------
    A LimeTabularExplainer summary of a randomly chosen
    example with label `label`. This is HTML displayed
    using IPython specials.
    """
    X = X.toarray()
    target_summary = next(d for d in summary if d['target'] == target)
    # The first of the models is as good as any, since
    # they were trained on random folds.
    mod = target_summary['models'][0]
    explainer = LimeTabularExplainer(
        X,
        feature_names=vectorizer.get_feature_names(), 
        class_names=mod.classes_,
        discretize_continuous=True)
    # TODO: Should take care not to sample an example that was used to
    # train `mod`, but we can set this aside for now, since we're 
    # basically just demoing LIME:
    index = np.random.choice([i for i, cls in enumerate(ys[target].values)
                              if cls==label])    
    exp = explainer.explain_instance(
        X[index],
        mod.predict_proba, 
        num_features=num_features, 
        top_labels=top_labels)
    exp.show_in_notebook(show_table=True, show_all=False)


def target_barplot(series, figsize):
    """Slightly specialized horizontal barplots, displaying
    raw counts with percentages as annotations.

    Parameters
    ----------
    series : pd.Series
        The column to plot.

    figsize : tuple of int
       The width and height in inches.

    Returns
    -------
    pyplot Axis
    """    
    ax = plt.subplot()
    counts = series.value_counts()
    vals = counts.values
    pers = vals / vals.sum()
    ax = plt.subplot()
    counts.plot(title=series.name, kind='barh', figsize=figsize, ax=ax)
    nudged_vals = vals * 1.01    
    for i, val in enumerate(nudged_vals):
        formatted_per = '{:0.01%}'.format(pers[i])
        ax.text(nudged_vals[i], i, formatted_per)
    ax.set_xlim((0, max(counts.values)*1.2))
    return ax


def specialty_mosaic(
        ys, cat2='gender',
        specialty_mincount=1500,
        specialty_maxcount=None):
    """Mosaic plot of the specialty variable and cat2, with the
    number of specialties controlled somewhat for intelligibility.

    Parameters
    ----------
    ys : a pd.DataFrame
        As produced by `build_experimental_dataset`

    cat2 : str
        The name of a column in `ys`.

    specialty_mincount : int
       Keep only specialties with at least many occurrences.

    specialty_maxcount : int or None
       Keep only specialties with at most many occurrences.
       If `None`, then this is set to the largest count in
       the data, equivalent to no imposed upper-bound.

    Use `plt.show()` to see the plot.
    """                     
    cat2 = ys[cat2]
    specialty = ys['specialty']
    dist = specialty.value_counts()
    if specialty_maxcount == None:
        specialty_maxcount = dist.max()    
    subset = [s for s,c in dist.items()
              if c >= specialty_mincount and c <= specialty_maxcount]        
    specialty = specialty[specialty.isin(subset)]
    target_mosaic(cat2, specialty)


def target_mosaic(cat1, cat2, figsize=(18, 4)):
    """Convenience function for creating a mosaic plot from
    `cat` and `cat2`.

    cat1 : pd.Series
       These values determine the y-axis.
       
    cat2 : pd.Series
       These values determine the x-axis.
       
    figsize : tuple
        The first coordinate is the width and the second is the        
        height, in inches.

    Use `plt.show()` to see the plot.        
    """
    xtab = pd.crosstab(cat1, cat2).unstack()
    # Bas colors:
    colors = ['#0499CC', '#4D8951', '#FDBA58', '#876DB5',
              '#32A8B4', '#9BB8D7', '#839A8A']        
    color_count = len(colors)
    # These need to be strings for `mosaic`.
    cat1_levels = list(map(str, cat1.value_counts().keys().values))
    cat2_levels = list(map(str, cat2.value_counts().keys().values))
        
    def prop(key):
        """Find the basse color for category c2 and then
        adjust it according to `c1`."""
        c2, c1 = key                       
        cat1_index = cat1_levels.index(c1)
        cat2_index = cat2_levels.index(c2)
        base_color = colors[cat2_index % color_count]
        adjusted = increase_luminance(base_color, multiplier=cat1_index)
        return adjusted

    # Display only the y-axis category inside the box:
    lab = (lambda key : key[1])
    fig, _ = mosaic(xtab, gap=0.01, labelizer=lab, properties=prop)
    figwidth, figheight = figsize
    fig.set_figwidth(figwidth)
    fig.set_figheight(figheight)    

    
def increase_luminance(color_str, multiplier=0):
    """Increase the luminance of the color represented by `color_str`.

    Parameters
    ----------
    color_str : str
        A color in one of the formats recognized by `colour.Color`.

    multipler : int
        The final color will be decreased by 0.1*multiplier, from
        a starting point of 0.8.

    Returns
    -------
    dict
        {'color': str} where str is the modified color. (This is the
        format required by `mosaic`.
    """
    
    c = Color(color_str)
    lum = 0.8 - np.repeat(0.1, multiplier).sum()
    c.luminance = lum
    return {'color': str(c)}



class DeepClassifier:
    """Defines a feed-forward neural network with two hidden layers.
    Roughly,

    h1 = f(xW1 + b1)
    h2 = g(h1W2 + b2)
    y = softmax(h2W3 + b3)

    where drop-out is applied to h1 and h2.    
    """
    def __init__(self,
            hidden_dim1=200,
            hidden_dim2=100,
            activation1=tf.nn.relu,
            activation2=tf.nn.relu,
            keep_prob1=0.7,
            keep_prob2=0.7,
            eta=0.01,
            max_iter=100,
            tol=1e-05,
            verbose=True):
        """
        Parameters
        ----------
        hidden_dim1 : int
            Dimensionality of the first hidden layer.

        hidden_dim2 : int
            Dimensionality of the second hidden layer.

        activation1 : tf.nn activation function
            Activation function for the first layer.

        activation1 : tf.nn activation function
            Activation function for the second layer.

        keep_prob1 : float
            Probability of keeping a unit in the first drop-out layer.

        keep_prob2 : float
            Probability of keeping a unit in the second drop-out layer.

        eta : float
            Learning rate.

        max_iter : t (default: 2000)
            Number of iterations of learning.

        tol : float (default: 0.0001)
            Tolerance for the loss. If it gets to this point,
            training stops even if `max_iter` has not been
            reached.

        verbose : bool
            Whether to report progress after each iteration.
        """
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.activation1 = activation1
        self.activation2 = activation2
        self.keep_prob1 = keep_prob1
        self.keep_prob2 = keep_prob2
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.params = ('hidden_dim1', 'hidden_dim2',
                       'activation1', 'activation2',
                       'keep_prob1', 'keep_prob2',                       
                       'eta', 'max_iter', 'tol',
                       'verbose')
        
    def fit(self, X, y):
        """Specifies the model graph and performs training.
        
        Parameters
        ----------        
        X : np.array, shape (n_examples, input_dim)
            The training matrix.
        y : array-like, shape (n_samples,)
            The labels for the rows/examples in `X`.
        
        Attributes
        ----------
        self.input_dim (int)
        self.output_dim (int)
        self.inputs : tf placeholder for input data
        self.outputs : tf placeholder for label matrices
        self.sess: TensorFlow interactive session
        self.dropout1 : tf placeholder for the first dropout layer
        self.dropout2 : tf placeholder for the second dropout layer
        """
        # Set-up the dataset:
        self.input_dim = X.shape[1]        
        self.classes_ = sorted(set(y))
        self.output_dim = len(self.classes_)
        y_ = self.onehot_encode(y)        
        # Begin the tf session:
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()        
        # Model:
        self._build_graph()        
        # Optimization:
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.model, self.outputs))
        optimizer = tf.train.AdagradOptimizer(self.eta).minimize(cost)
        # Initialization:
        init = tf.initialize_all_variables()
        self.sess.run(init)                        
        for i in range(1, self.max_iter+1):
            # Training step:
            _, loss = self.sess.run([optimizer, cost],
                feed_dict={self.inputs: X,
                           self.outputs: y_,
                           self.keepprob_holder1: self.keep_prob1,
                           self.keepprob_holder2: self.keep_prob2})
            # Progress report:
            self._progressbar(loss, i)
            if loss <= self.tol:
                sys.stderr.write('Stopping criteria reached.')
        if self.verbose:
            sys.stderr.write("\n")

    def _build_graph(self):
        """Builds the core computation graph."""
        # Inputs and outputs:
        self.inputs = tf.placeholder(tf.float32, [None, self.input_dim])
        self.outputs = tf.placeholder(tf.float32, [None, self.output_dim])
        # Layer 1:        
        W1 = self._weight_init(self.input_dim, self.hidden_dim1, name='W1')
        b1 = self._bias_init(self.hidden_dim1, name='b1')
        hidden1 = self.activation1(tf.matmul(self.inputs, W1) + b1)
        # Dropout 1:
        self.keepprob_holder1 = self._dropout_init('keep_prob1')
        dropout_layer1 = tf.nn.dropout(hidden1, self.keepprob_holder1)
        # Layer 2:
        W2 = self._weight_init(self.hidden_dim1, self.hidden_dim2, name='W2')
        b2 = self._bias_init(self.hidden_dim2, name='b2')
        hidden2 = self.activation2(tf.matmul(dropout_layer1, W2) + b2)
        # Dropout 2:
        self.keepprob_holder2 = self._dropout_init('keep_prob2')
        dropout_layer2 = tf.nn.dropout(hidden2, self.keepprob_holder2)
        # Output layer:
        W3 = self._weight_init(self.hidden_dim2, self.output_dim, name='W3')
        b3 = self._bias_init(self.output_dim, name='b3')
        # No softmax here; that's handled by the cost function.
        self.model = tf.matmul(dropout_layer2, W3) + b3

    def predict(self, X):
        """Predict method that mimics `sklearn` by accepting
        an np.array and returning a vector of predictions.

        X : np.array (must be dense)
            Features for prediction.

        Returns
        -------
        np.array
           Predictions. This is the list of highest probability
           categories, rather than the full matrix of probabilities
           used internally.
        """
        predictions = self.sess.run(self.model,
            feed_dict={self.inputs: X,
                       self.keepprob_holder1: 1.0,
                       self.keepprob_holder2: 1.0})                                    
        return self._predictionvecs2class(predictions)
    
    def _weight_init(self, m, n, name):
        """Weight initialization according to the heuristic
        that the values should be uniformly distributed around
        
        +/sqrt(6/(m+n))

        where m is the dimension of the incoming layer (fan in) and
        n is the dimension of the outgoing layer (fan out). This
        could be replaced by `tf.contrib.layers.xavier_initializer`.
            
        Parameters
        ----------
        m : int
            The dimensionality of the incoming layer.
    
        n : int
            The dimensionality of the outgoing layer.

        name : str
            Name for the variable, for TensorBoard visualization.

        Returns
        -------
        tf.Variable
            Tensor of random floats in the desired range, shape (m,n).
        """
        x = np.sqrt(6.0/(m+n))
        with tf.name_scope(name) as scope: 
            return tf.Variable(
                tf.random_uniform(
                    [m, n], minval=-x, maxval=x), name=name)

    def _bias_init(self, dim, name, constant=0.0):
        """Bias initialization, by default as all 0s.
        
        Parameters
        ----------
        dim : int
            The dimension of the resulting vector (`tf.Variable`).

        name : str
            Name for the variable, for TensorBoard visualization.
        
        constant : float (default: 0.0)
            The constant value for the bias. Some models benefit
            from a small constant value for their bias.

        Returns
        -------
        tf.Variable
            The bias vector.        
        """
        with tf.name_scope(name) as scope:            
            return tf.Variable(
                tf.constant(constant, shape=[dim]), name=name)

    def _dropout_init(self, name):
        """Initialize a placeholder for a dropout value."""        
        with tf.name_scope(name) as scope:
            return tf.placeholder(tf.float32, name=name)
                
    def onehot_encode(self, y, on_value=1.0):
        """Turns the list of class labels `y` into a matrix of
        one-hot encoded vectors. This could be replaced by
        `tf.one_hot`, but this native version does the job.
        
        Parameters
        ----------
        y : array-like, shape (n_samples,)
            The labels to encode.

        on_value : float
            Value to use for `on` value. Defaults to 1.0.
        
        Returns
        -------
        np.array, shape (n_examples, n_classes)
            Each row is all 0s except for a 1 in the position of the
            class label according to the ordering in `self.classes_`.
        """        
        classmap = dict(zip(self.classes_, range(self.output_dim)))        
        y_ = np.zeros((len(y), self.output_dim))
        for i, cls in enumerate(y):
            y_[i][classmap[cls]] = on_value            
        return y_

    def _progressbar(self, loss, index):
        """Overwriting progress bar for feedback on training process.
        Prints to standard error.

        Parameters
        ----------
        loss : float
            Current training error
        index :
            The index of the current training iteration.
        """        
        if self.verbose:        
            sys.stderr.write('\r')
            sys.stderr.write("Iteration {}: loss is {}".format(index, loss))
            sys.stderr.flush()

    def get_params(self, deep=True):
        """Gets the hyperparameters for the model, as given by the
        `self.params` attribute. This is called `get_params` for
        compatibility with sklearn. `deep=True` is ignored, but is
        needed for sklearn.
        
        Returns
        -------
        dict
            Map from attribute names to their values.
        """
        return {p: getattr(self, p) for p in self.params}

    def set_params(self, **params):
        """Use the params dict to set attribute values. This
        is needed for sklearn `GridSearchCV` compatibility.

        Returns
        -------
        self
        """        
        for key, val in six.iteritems(params):
            setattr(self, key, val)
        return self

    def _predictionvecs2class(self, predictions):
        """
        Convert the matrix of prediction probabilities into classes.
        In cases of ties, a random choices is made to avoid spurious
        patterns resulting from guessing classes that are earlier
        in ordering in case of ties.
        
        Parameters
        ----------
        predictions : np.array, shape (n_examples, n_classes)
            The probabilistic predictions of the classifiers.
            
        Returns
        -------
        list
            The list of predicted class labels.                
        """
        maxprobs = predictions.max(axis=1)
        cats = []
        for row, maxprob in zip(predictions, maxprobs):
            i = np.random.choice([i for i, val in enumerate(row)
                                  if val==maxprob])
            cats.append(self.classes_[i])        
        return cats
    
