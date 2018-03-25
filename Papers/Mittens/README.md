# Mittens: An Extension of GloVe for Learning Domain-Specialized Representations

Nick Dingwall and Christopher Potts

Supplementary materials for NAACL 2018 short paper.

All this code is written in Python 3:

* Our vectorized implementations of GloVe and Mittens can be obtained with `pip install mittens`.
* `speed_tests.ipynb` runs the speed tests reported in table 1.
* `nonvectorized_glove.py` is the _Nonvectorized Tensorflow_ implementation used in the speed tests.
* `mittens_simulations.ipynb` runs the simulations for figure 1.
* `imdb_sentiment.ipynb` runs the experiments for section 3.
* Supporting code is in `tokenizing.py` and `utils.py`
* `bootstrap.py` is from https://github.com/cgevans/scikits-bootstrap and is included here for reproducibility.
* The `results` directory contains the output of all evaluations.

Unfortunately, we are not able to redistribute the word representations we learned from clinical text (section 4).
