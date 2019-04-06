""" We use the Drug-Disease Relations dataset here as an example to show how to
train the four models with (readily tuned) hyperparameters and evaluate
the performance of models on the test set.

An example to run the demo:
  python run_model.py --model_name combined

where the model is specified by the parameter "model_name". It can be one of
the four options:
  "rand_lstm_crf": rand-LSTM-CRF
  "elmo_lstm_crf": ELMo-LSTM-CRF
  "hb_crf":        HB-CRF
  "combined":      ELMo-LSTM-CRF-HB
"""

import os
import argparse
from collections import defaultdict as dd
import pickle

from utils import bool_ext, set_default_args, get_ddr_dataset
from model import model

__author__ = "Chris Potts and Yifeng Tao"


parser = argparse.ArgumentParser()

parser.add_argument("--model_name", help="name of model", type=str, default="rand_lstm_crf")
parser.add_argument("--default_config", help="whether to use default tuned configuration parameters", type=bool_ext, default=True)

parser.add_argument("--max_iter", help="samples to train over during training", type=int, default=0)
parser.add_argument("--eta", help="training step size", type=float, default=0)
parser.add_argument("--batch_size", help="traning batch size", type=int, default=16)
parser.add_argument("--test_batch_size", help="test batch size", type=int, default=32)
parser.add_argument("--hid_dim", help="dimension of LSTM hidden layer", type=int, default=50)
parser.add_argument("--hid_hb_dim", help="dimension of dense hand-built feature", type=int, default=100)
parser.add_argument("--emb_dim", help="dimension of word embedding", type=int, default=3072)
parser.add_argument("--max_len", help="maximum length of sentences", type=int, default=600)
parser.add_argument("--c1", help="l1 regularizer coefficient", type=float, default=0)
parser.add_argument("--c2", help="l2 regularizer coefficient", type=float, default=0)

parser.add_argument("--inc", help="number of training steps for every evaluation", type=int, default=64)

parser.add_argument("--input_dir", help="directory of input files", type=str, default="roam_drug_disease_relations")
parser.add_argument("--output_dir", help="directory of output files", type=str, default="output")

args = parser.parse_args()


assert args.model_name in ["rand_lstm_crf", "hb_crf", "elmo_lstm_crf", "combined"]
assert args.inc % args.batch_size == 0

# Get default tuned configurations of models, such as step size, minibatch size etc.
if args.default_config:
  assert args.batch_size == 16
  assert args.test_batch_size == 32
  assert args.hid_dim == 50
  assert args.hid_hb_dim == 100
  assert args.input_dir == "roam_drug_disease_relations"
  assert args.output_dir == "output"
  args = set_default_args(args.model_name, args)

if not os.path.exists(args.output_dir):
  os.makedirs(args.output_dir)

# Path to save results.
result_dir = os.path.join(args.output_dir, args.model_name)
if not os.path.exists(result_dir):
  os.makedirs(result_dir)

# Print experiment configurations.
keys = list(args.__dict__.keys())
keys.sort()
strings = "="*64+"\n"+"\n".join([k+"="+str(args.__dict__[k]) for k in keys])+"\n"+"_"*64
print(strings)
with open(os.path.join(result_dir,"log.txt"), "a") as f:
  f.write(strings+"\n")

# Load the training and test sets of Drug-Disease Relations dataset (DDR).
X_train, X_test, X_elmo_train, X_elmo_test, \
X_sparse_train, X_sparse_test, y_train, \
y_test, feat2idx, vocab = get_ddr_dataset(args)

# Record the ground truth and model predictions.
results = dd(dd)
results["X_test"] = X_test
results["y_test"] = y_test

# Train the model.
rnn = model(
    vocab=vocab,
    model_name=args.model_name,
    max_iter=args.max_iter,
    eta=args.eta,
    batch_size=args.batch_size,
    test_batch_size=args.test_batch_size,
    hid_dim=args.hid_dim,
    hid_hb_dim=args.hid_hb_dim,
    emb_dim=args.emb_dim,
    feat_dim=len(feat2idx),
    max_len=args.max_len,
    feat2idx=feat2idx,
    c1=args.c1,
    c2=args.c2)
rnn.fit(X_train, X_elmo_train, X_sparse_train, y_train,
        X_test=X_test, X_elmo_test=X_elmo_test,
        X_sparse_test=X_sparse_test, y_test=y_test,
        inc=args.inc, log=os.path.join(result_dir,"log.txt"))

# Record evaluations on test set.
results["preds"] = rnn.preds
results["eval"] = rnn.eval_test
results["f1_macro"] = rnn.eval_test["f1_macro"]

with open( os.path.join(result_dir, "results.pkl"), "wb") as f:
  pickle.dump(results, f)

