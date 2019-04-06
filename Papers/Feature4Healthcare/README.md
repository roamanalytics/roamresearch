# Effective Feature Representation for Clinical Text Concept Extraction

## Introduction

This repository contains experiments code and TensorFlow implementation of
models (rand-LSTM-CRF, HB-CRF, ELMo-LSTM-CRF, and ELMo-LSTM-CRF-HB) for
clinical named-entity recognition (NER) in the following paper:
Yifeng Tao, Bruno Godefroy, Guillaume Genthial, and Christopher Potts.
**[Effective Feature Representation for Clinical Text Concept Extraction](https://arxiv.org/abs/1811.00070)**.
*Proceedings of the NAACL Workshop on Clinical Natural Language Processing (NAACL-ClinicalNLP)*, 2019.

The Drug-Disease Relations dataset \[[link](https://github.com/roamanalytics/roamresearch/tree/master/BlogPosts/Features_for_healthcare)\],
which is generated using crowdscourcing and curated by a team of experts is released together with the paper as well.

## Prerequisites

The code runs on `Python 3.6` and `TensorFlow 1.10`.
`ELMo` is required to extract ELMo embeddings.
The following Python packages are required as well:
`nltk`, `sklearn-crfsuite`, `h5py`, `pickle`, `numpy`, `json`, `random`, `collections`, `argparse`, `string`, `operator`, `os`.

## Preprocessing

We may need to download the dataset, tokenize, featurize and extract ELMo embeddings
before running experiments.

### Download Code and Dataset

Download the source code and Drug-Disease Relations dataset:
```
git clone https://github.com/roamanalytics/roamresearch.git
cd roamresearch/Papers/Features4Healthcare
cp ../../BlogPosts/Features_for_healthcare/roam_drug_disease_relations_dataset.zip ./
unzip roam_drug_disease_relations_dataset.zip
```

### Tokenize and Featurize Sentences

Tokenize, lemmatize, and featurize sentences to get sparse hand-built features of Drug-Disease Relations dataset:
```
python prepare_sparse_features.py
```

### Extract ELMo Embeddings

Download and install ELMo under the directory `Feature4Healthcare`:
```
git clone https://github.com/allenai/bilm-tf.git
mv prepare_elmo_features.py bilm-tf/prepare_elmo_features.py
cd bilm-tf
python setup.py install
```

Download the weights \[[link](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)\]
and options \[[link](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json)\]
files of ELMo under directory `bilm-tf`.

Extract dense EMLo embeddings (takes around 1.5 hours on a desktop computer):
```
python prepare_elmo_features.py
cd ../
```

## Run Experiments

`models.py` contains implementation of four models.
To train and evaluate the models with tuned hyperparameters on Drug-Disease Relations dataset:
```
python run_model.py --model_name [MODEL_NAME]
```
The model is specified by `[MODEL_NAME]`, which can be `rand_lstm_crf` (rand-LSTM-CRF),
`hb_crf` (HB-CRF), `elmo_lstm_crf` (ELMo-LSTM-CRF), `combined` (EMLo-LSTM-CRF-HB). E.g.,
to run the EMLo-LSTM-CRF-HB model:
```
python run_model.py --model_name combined
```

To see more options:
```
python run_model.py --help
```

## Citation

Please cite this paper if you use the code from this repo or Drug-Disease Relations dataset:
```
@article{tao2019lstmcrf,
  title = {Effective Feature Representation for Clinical Text Concept Extraction},
  author = {Tao, Yifeng and Godefroy, Bruno and Genthial, Guillaume and Potts, Christopher},
  journal = {NAACL Workshop on Clinical Natural Language Processing},
  year = {2019}
}
```
