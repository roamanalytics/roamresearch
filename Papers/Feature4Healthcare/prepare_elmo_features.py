import os
import h5py

from bilm import dump_bilm_embeddings

__author__ = "Yifeng Tao"


def prepare_elmo_features(path, dataset, vocab_file, options_file, weight_file):
  """ Dump the embeddings to a file.

  Parameters
  ----------
  path: str
  dataset: str
  vocab_file: str
  options_file: str
  weight_file: str

  """

  embedding_file = os.path.join(path, "X_elmo_"+dataset+".hdf5")
  dump_bilm_embeddings(vocab_file, dataset_file, options_file, weight_file, embedding_file)


path = "../roam_drug_disease_relations"

vocab_file = "../roam_drug_disease_relations/vocab.txt"
options_file = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

for dataset in ["train", "test"]:
  dataset_file = "../roam_drug_disease_relations/tokens_"+dataset+".txt"
  print("Preparing ELMo features for "+dataset+" set...")
  prepare_elmo_features(path, dataset, vocab_file, options_file, weight_file)

