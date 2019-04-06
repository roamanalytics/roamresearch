import os
import operator
import string
from random import shuffle
from collections import Counter

import json
import pickle

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer

__author__ = "Chris Potts and Yifeng Tao"


def tokenize(data):
  """ Tokenization.
  Tokenize and lemmatize the sentences; extract labels of tokens.

  Parameters
  ----------
  data : list of dict
    each dict should have the following form:
    {"sentence": str,
    "sentence_id": str,
    "annotations": [
      {"ann_id": str
	      "text": str,
	      "start": int,
	      "end": int,
	      "label": str
      }}

  Returns
  -------
  list_tokens
  list_labels
  """
  tknzr = TweetTokenizer()
  lemmatizer = WordNetLemmatizer()

  list_tokens, list_labels = [], []

  for idx in range(len(data)):
    sample = data[idx]
    sent = sample["sentence"]
    tokens = tknzr.tokenize(sent)

    lem_tokens = [lemmatizer.lemmatize(t) for t in tokens]
    lem_tokens = ["".join([t if ord(t) < 128 else "*" for t in list(token)])
      for token in lem_tokens]

    idx_char = 0
    labels = []
    for t in tokens:
      label = "Other"
      while t != sent[idx_char:idx_char+len(t)]:
        idx_char += 1
      for ann in sample["annotations"]:
        if (ann["start"] <= idx_char) and (idx_char+len(t) <= ann["end"]):
          label = ann["label"]
      idx_char += len(t)
      labels.append(label)

    list_tokens.append(lem_tokens)
    list_labels.append(labels)

  return list_tokens, list_labels


def featurize(list_tokens):
  """ Featurization.
  Extract the sparse hand-built features from tokenized sentences.
  Note that the hand-built features here are much simpler than the ones we used
  in the paper.

  Parameters
  ----------
  list_tokens : list of list of str
    each str is a lemmatized token

  Returns
  -------
  list_sent_feature : list of list of dict
    each dict is the features for a token
  """

  list_sent_feature = []

  for idx, tokens in enumerate(list_tokens):
    sent_feature = []
    tagged = nltk.pos_tag(tokens)
    tokens = [str(t) for t in tokens]
    for idx_word, word in enumerate(tokens):
      word_feature = {
          "bias":1.0,
          "tag":tagged[idx_word][1],
          "word":word,
          "is_upper":word.isupper(),
          "is_title":word.istitle(),
          "is_punctuation":False}

      if (len(word) == 1) and (word in string.punctuation):
        word_feature["is_punctuation"] = True

      for idx_p in range(1, 5):
        if idx_word >= idx_p:
          word_feature["word-"+str(idx_p)] = tokens[idx_word-idx_p]
          word_feature["tag-"+str(idx_p)] = tagged[idx_word-idx_p][1]
        if idx_word <= len(tokens)-1-idx_p:
          word_feature["word+"+str(idx_p)] = tokens[idx_word+idx_p]
          word_feature["tag+"+str(idx_p)] = tagged[idx_word+idx_p][1]
      sent_feature.append(word_feature)

    list_sent_feature.append(sent_feature)

  return list_sent_feature


def prepare_sparse_features(path):
  """ Tokenize and repare the sparse features of dataset.
  The training.json and test.json files are loaded and the results are saved
  under the same directory.

  """

  print("Tokenizing...")
  with open(os.path.join(path, "training.json"), "r") as f:
    sentences = json.load(f)
  shuffle(sentences)
  X_train, y_train = tokenize(sentences)

  with open(os.path.join(path, "test.json"), "r") as f:
    sentences = json.load(f)
  shuffle(sentences)
  X_test, y_test = tokenize(sentences)

  print("Featurizing...")
  X_sparse_train = featurize(X_train)
  X_sparse_test = featurize(X_test)

  print("Indexing sparse features...")
  # Map feature name to feature index.
  tmp = ["str_"+k+"_"+word[k] if type(word[k]) is str
         else "float_"+k for sent in (X_sparse_train + X_sparse_test)
         for word in sent for k in word.keys()]
  features = list(set(tmp))
  features.sort()
  feat2idx={feat:idx for idx, feat in enumerate(features)}

  data_sparse = {
      "X_train":X_train,
      "X_test":X_test,
      "X_sparse_train":X_sparse_train,
      "X_sparse_test":X_sparse_test,
      "y_train":y_train,
      "y_test":y_test,
      "feat2idx":feat2idx}

  with open(os.path.join(path, "data_sparse.pkl"), "wb") as f:
    pickle.dump(data_sparse, f)

  with open(os.path.join(path, "tokens_train.txt"), "w") as f:
    f.write("\n".join([" ".join(sent) for sent in X_train]))

  with open(os.path.join(path, "tokens_test.txt"), "w") as f:
    f.write("\n".join([" ".join(sent) for sent in X_test]))

  print("Preparing vocabulary...")
  words = [word for sent in (X_train+X_test) for word in sent]
  word2count = Counter(words)
  word_and_count = sorted(word2count.items(),key=operator.itemgetter(1),reverse=True)
  vocab = [wc[0] for wc in word_and_count]
  vocab = ["<S>", "</S>"] + vocab
  with open(os.path.join(path, "vocab.txt"), "w") as f:
    f.write("\n".join(vocab))

path = "roam_drug_disease_relations"
prepare_sparse_features(path)

