"""utils.py

Shared utilities for the code for reproducing results from Dingwall
and Potts, 'Mittens: an extension of glove for learning domain-
specialized representations' (NAACL 2018).
"""
import bootstrap
from collections import Counter, defaultdict
import csv
import numpy as np
from operator import itemgetter
import os
import pandas as pd
import random
from scipy.stats import pearsonr
from sklearn_crfsuite.metrics import (
    flat_classification_report, flat_f1_score,
    flat_precision_score, flat_recall_score,
    flat_accuracy_score, sequence_accuracy_score)
from tokenizing import WordOnlyTokenizer

__author__ = 'Nick Dingwall and Christopher Potts'


TOKENIZER = WordOnlyTokenizer(lower=True, preserve_acronyms=True)


def basic_tokenizer(s):
    if isinstance(s, str):
        return TOKENIZER.tokenize(s)
    else:
        return []


def build_weighted_matrix(corpus, tokenizing_func=basic_tokenizer,
        mincount=300, vocab_size=None, window_size=10,
        weighting_function=lambda x: 1 / (x + 1)):
    """Builds a count matrix based on a co-occurrence window of
    `window_size` elements before and `window_size` elements after the
    focal word, where the counts are weighted based on proximity to the
    focal word.

    Parameters
    ----------
    corpus : iterable of str
        Texts to tokenize.
    tokenizing_func : function
        Must map strings to lists of strings.
    mincount : int
        Only words with at least this many tokens will be included.
    vocab_size : int or None
        If this is an int above 0, then, the top `vocab_size` words
        by frequency are included in the matrix, and `mincount`
        is ignored.
    window_size : int
        Size of the window before and after. (So the total window size
        is 2 times this value, with the focal word at the center.)
    weighting_function : function from ints to floats
        How to weight counts based on distance. The default is 1/d
        where d is the distance in words.

    Returns
    -------
    pd.DataFrame
        This is guaranteed to be a symmetric matrix, because of the
        way the counts are collected.

    """
    tokens = [tokenizing_func(text) for text in corpus]

    # Counts for filtering:
    wc = defaultdict(int)
    for toks in tokens:
        for tok in toks:
            wc[tok] += 1
    if vocab_size:
        srt = sorted(wc.items(), key=itemgetter(1), reverse=True)
        vocab_set = {w for w, c in srt[: vocab_size]}
    else:
        vocab_set = {w for w, c in wc.items() if c >= mincount}
    vocab = sorted(vocab_set)
    n_words = len(vocab)

    # Weighted counts:
    counts = defaultdict(float)
    for toks in tokens:
        window_iter = _window_based_iterator(toks, window_size, weighting_function)
        for w, w_c, val in window_iter:
            if w in vocab_set and w_c in vocab_set:
                counts[(w, w_c)] += val

    # Matrix:
    X = np.zeros((n_words, n_words))
    for i, w1 in enumerate(vocab):
        for j, w2 in enumerate(vocab):
            X[i, j] = counts[(w1, w2)]

    # DataFrame:
    X = pd.DataFrame(X, columns=vocab, index=pd.Index(vocab))
    return X


def _window_based_iterator(toks, window_size, weighting_function):
    for i, w in enumerate(toks):
        yield w, w, 1
        left = max([0, i - window_size])
        for x in range(left, i):
            yield w, toks[x], weighting_function(abs(x - i))
        right = min([i + 1 + window_size, len(toks)])
        for x in range(i+1, right):
            yield w, toks[x], weighting_function(abs(x - i))


def sequence_length_report(X, potential_max_length=50):
    lengths = [len(ex) for ex in X]
    longer = len([x for x in lengths if x > potential_max_length])
    print("Max sequence length: {:,}".format(max(lengths)))
    print("Min sequence length: {:,}".format(min(lengths)))
    print("Mean sequence length: {:0.02f}".format(np.mean(lengths)))
    print("Median sequence length: {:0.02f}".format(np.median(lengths)))
    print("Sequences longer than {:,}: {:,} of {:,}".format(
            potential_max_length, longer, len(lengths)))


def evaluate_rnn(y, preds):
    """Because the RNN sequences get clipped as necessary based
    on the `max_length` parameter, they have to be realigned to
    get a classification report. This method does that, building
    in the assumption that any clipped tokens are assigned an
    incorrect label.

    Parameters
    ----------
    y : list of list of labels
    preds : list of list of labels

    Both of these lists need to have the same length, but the
    sequences they contain can vary in length.
    """
    labels = sorted({c for ex in y for c in ex})
    new_preds = []
    for gold, pred in zip(y, preds):
        delta = len(gold) - len(pred)
        if delta > 0:
            # Make a *wrong* guess for these clipped tokens:
            pred += [random.choice(list(set(labels)-{label}))
                     for label in gold[-delta: ]]
        new_preds.append(pred)
    labels = sorted({cls for ex in y for cls in ex} - {'OTHER'})
    data = {}
    data['classification_report'] = flat_classification_report(y, new_preds)
    data['f1_macro'] = flat_f1_score(y, new_preds, average='macro')
    data['f1_micro'] = flat_f1_score(y, new_preds, average='micro')
    data['f1'] = flat_f1_score(y, new_preds, average=None)
    data['precision_score'] = flat_precision_score(y, new_preds, average=None)
    data['recall_score'] = flat_recall_score(y, new_preds, average=None)
    data['accuracy'] = flat_accuracy_score(y, new_preds)
    data['sequence_accuracy_score'] = sequence_accuracy_score(y, new_preds)
    return data


def get_random_rep(embedding_dim=50, scale=0.62):
    """The `scale=0.62` is derived from study of the external GloVE
    vectors. We're hoping to create vectors with similar general
    statistics to those.
    """
    return np.random.normal(size=embedding_dim, scale=0.62)


def dataframe2classifier_embedding(filename, vocab, embedding_dim):
    df = pd.read_csv(filename, index_col=0)
    return np.array([df.loc[w].values if w in df.index else get_random_rep()
                     for w in vocab])


def create_random_lookup(vocab):
    """Create random representations for all the words in `vocab`,
    and return new random representstions for new words tha we
    try to look-up, adding them to the lookup when this happens.
    """
    data =  {w: get_random_rep() for w in vocab}
    return defaultdict(lambda : get_random_rep(), data)


def glove2dict(glove_filename):
    with open(glove_filename) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        data = {line[0]: np.array(list(map(float, line[1: ]))) for line in reader}
    return data


def create_glove_lookup(glove_filename):
    """Turns an external GloVe file into a defaultdict that returns
    the learned representation for words in the vocabulary and
    random representations for all others.
    """
    glove_lookup = glove2dict(glove_filename)
    glove_lookup = defaultdict(lambda : get_random_rep(), glove_lookup)
    return glove_lookup


def create_lookup(X):
    """Map a dataframe to a lookup that returns random vector reps
    for new words, adding them to the lookup when this happens.
    """
    embedding_dim = X.shape[1]
    data = defaultdict(lambda : get_random_rep())
    for w, vals in X.iterrows():
        data[w] = vals.values
    return data


def get_ci(vals):
    """Bootstrapped 95% confidence intervals."""
    return bootstrap.ci(vals, method='bca')


def correlation_test(true, pred):
    """Tests the extent to which w_i^Tw_j is proportional to various
    notions of word probability; see (6) and (7) in the GloVe paper.

    Parameters
    ----------
    true : np.array
        The count matrix.
    pred : np.array
        The learned representations.

    Returns
    -------
    dict giving the corrrelations between the dot product of
    learned vectors and

    * log_cooccur: log(X[i,j])
    * prob: log(X[i,j]) - log(X[i])
    * pmi: log(X[i,j]) - log(X[i]*X[j])
    """
    mask = true > 0
    M = pred.dot(pred.T)
    with np.errstate(divide='ignore'):
        log_cooccur = np.log(true)
        log_cooccur[np.isinf(log_cooccur)] = 0.0
        row_prob = np.log(true.sum(axis=1))
        col_prob = np.log(true.sum(axis=0))
        prob = log_cooccur - np.log(np.outer(row_prob, np.ones(true.shape[1])))
        pmi = log_cooccur - np.log(np.outer(row_prob, col_prob))
    data = {}
    for typ, val in (('log_cooccur',  log_cooccur),
                     ('prob', prob),
                     ('pmi', pmi)):
        rho = np.corrcoef(val[mask], M[mask])[0, 1]
        data[typ] = rho
    return data
