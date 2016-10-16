Optimizing the hyperparameter of which hyperparameter optimizer to use
==========

Authors: Ben Bernstein and Chris Potts

## Overview

This folder contains the supporting notebook and code for the Roam blog post
[Optimizing the hyperparameter of which hyperparameter optimizer to use](http://roamanalytics.com/2016/09/15/optimizing-the-hyperparameter-of-which-hyperparameter-optimizer-to-use/).

## Installation

This code has a variety of unusual requirements; you'll probably want to work with it inside a virtual environment.

* The `requirements.txt` file lists requirements in the usual way. Those packages should be installed __before__ the following steps are taken.

* We use the brand-new `scikit-optimize` package, which requires an as-yet-unreleased `sklearn`. To install this version:

  `pip install git+https://github.com/scikit-learn/scikit-learn.git#egg=scikit-learn-0.18dev`

* Install scikit-optimize:

  `pip install scikit-optimize==0.1`

* A bug in the released version of `hyperopt` means that it doesn't work under Python 3. However, installing directly from the Github repository works fine:

  `pip install git+https://github.com/hyperopt/hyperopt.git`

* XGBoost needs to be installed separately using the instructions here:

  https://xgboost.readthedocs.io/en/latest/build.html

* After XGBoost is installed, the Python package can be installed using these instructions:

  https://xgboost.readthedocs.io/en/latest/build.html#python-package-installation
