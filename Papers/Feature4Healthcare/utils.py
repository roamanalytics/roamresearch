""" Shared utilities for run_model.py and model.py.
"""
import os
import random
import numpy as np
import pickle
import h5py

from sklearn_crfsuite.metrics import (
    flat_classification_report, flat_f1_score,
    flat_precision_score, flat_recall_score,
    flat_accuracy_score, sequence_accuracy_score)

__author__ = "Chris Potts and Yifeng Tao"


def bool_ext(rbool):
  """ Solve the problem that raw bool type is always True.

  Parameters
  ----------
  rbool: str
    Should be True of False.
  """

  if rbool not in ["True", "False"]:
    raise ValueError("Not a valid boolean string")

  return rbool == "True"


def evaluate_rnn(y, preds):
  """ Evaluate the RNN performance using various metrics.

  Parameters
  ----------
  y: list of list of labels
  preds: list of list of labels

  Both of these lists need to have the same length, but the
  sequences they contain can vary in length.

  Returns
  -------
  data: dict
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
  labels = sorted({cls for ex in y for cls in ex} - {"OTHER"})
  data = {}
  data["classification_report"] = flat_classification_report(y, new_preds, digits=3)
  data["f1_macro"] = flat_f1_score(y, new_preds, average="macro")
  data["f1_micro"] = flat_f1_score(y, new_preds, average="micro")
  data["f1"] = flat_f1_score(y, new_preds, average=None)
  data["precision_score"] = flat_precision_score(y, new_preds, average=None)
  data["recall_score"] = flat_recall_score(y, new_preds, average=None)
  data["accuracy"] = flat_accuracy_score(y, new_preds)
  data["sequence_accuracy_score"] = sequence_accuracy_score(y, new_preds)

  return data


def set_default_args(model_name, args):
  """ Return the default (tuned) hyperparams of Drug-Disease Relations dataset.

  Parameters
  ----------
  model_name: str
    4 possible models in the paper:
      "rand_lstm_crf": rand-LSTM-CRF
      "elmo_lstm_crf": ELMo-LSTM-CRF
      "hb_crf":        HB-CRF
      "combined":      ELMo-LSTM-CRF-HB

  Returns
  -------
  args: dict
  """

  args.max_len = 544
  if model_name in ["elmo_lstm_crf", "hb_crf", "combined"]:
    args.max_iter = 4096*80
  elif model_name == "rand_lstm_crf":
    args.max_iter = 4096*300

  if model_name == "rand_lstm_crf":
    args.eta = 1e-4
    args.c1 = 3e-5
    args.c2 = 3e-3
    args.emb_dim = 50
  elif model_name == "elmo_lstm_crf":
    args.eta = 5e-6
    args.c1 = 0
    args.c2 = 1e-3
    args.emb_dim = 3072
  elif model_name == "hb_crf":
    args.eta = 1e-4
    args.c1 = 3e-6
    args.c2 = 1e-4
    args.emb_dim = 3072
  elif model_name == "combined":
    args.eta = 1e-5
    args.c1 = 3e-7
    args.c2 = 3e-5
    args.emb_dim = 3072

  return args


def get_ddr_dataset(args):
  """ Return Drug-Disease Relations (DDR) dataset (training and test sets).
  It loads tokenized sentences, ELMo embeddings, hand-built sparse features and labels.
  We do not shuffle training set here since it is done in prepare_sparse_features.py.

  Parameters
  ----------
  args: dict

  Returns
  -------
  X_train, X_test: list of list of str
      Tokenized sentences.
  X_elmo_train, X_elmo_test: list
      ELMo embedding of each token in sentences.
  X_sparse_train, X_sparse_test: list of list of dict
      Same format input as that of sklearn_crfsuite.
  y_train, y_test: list of list of str
      Ground truth labels of tokens.
  feat2idx: dict
      Feature name to index of the hand-built feature.
  vocab: list of str
      List of tokenized words contained in the dataset.
  """

  # Ground truth labels
  print("Loading dataset...")
  with open(os.path.join(args.input_dir, "data_sparse.pkl"), "rb") as f:
    data_sparse = pickle.load(f)
  y_train = data_sparse["y_train"]
  y_test = data_sparse["y_test"]

  # Tokenized sentences
  X_train = data_sparse["X_train"]
  X_test = data_sparse["X_test"]

  # Vocabulary
  vocab_train = set([word for sent in X_train for word in sent])
  vocab_test = set([word for sent in X_test for word in sent])
  vocab = list(vocab_train | vocab_test)

  size_train_set = len(y_train)
  size_test_set = len(y_test)
  args.max_len = max(args.max_len, max([len(yy) for yy in (y_train+y_test)]) )

  # Hand-built sparse features.
  if args.model_name in ["hb_crf", "combined"]:
    print("Loading hand-built sparse features...")
    X_sparse_train = data_sparse["X_sparse_train"]
    X_sparse_test = data_sparse["X_sparse_test"]
    feat2idx = data_sparse["feat2idx"]
  else: #args.model_name in ["rand_lstm_crf", "elmo_lstm_crf"]
    X_sparse_train = [0]*size_train_set
    X_sparse_test = [0]*size_test_set
    feat2idx = {}

  # ELMo dense embedding features
  if args.model_name in ["elmo_lstm_crf", "combined"]:
    print("Loading ELMo embeddings...")
    X_elmo_train = np.zeros( (size_train_set,args.max_len,3072) ,dtype=float)
    with h5py.File(os.path.join(args.input_dir, "X_elmo_train.hdf5"), "r") as f:
      for i in range(len(f.keys())):
        embeddings = f[str(i)][...]
        tmp = np.concatenate((embeddings[0], embeddings[1], embeddings[2]), axis=1)
        tmp = tmp[:args.max_len]
        X_elmo_train[i,0:len(tmp)] = tmp
    X_elmo_test = np.zeros( (size_test_set,args.max_len,3072) ,dtype=float)
    with h5py.File(os.path.join(args.input_dir, "X_elmo_test.hdf5"), "r") as f:
      for i in range(len(f.keys())):
        embeddings = f[str(i)][...]
        tmp = np.concatenate((embeddings[0], embeddings[1], embeddings[2]), axis=1)
        tmp = tmp[:args.max_len]
        X_elmo_test[i,0:len(tmp)] = tmp
  else: #args.model_name in ["rand_lstm_crf", "hb_crf"]
    X_elmo_train = [0]*size_train_set
    X_elmo_test = [0]*size_test_set

  return X_train, X_test, X_elmo_train, X_elmo_test, \
      X_sparse_train, X_sparse_test, y_train, y_test, feat2idx, vocab

