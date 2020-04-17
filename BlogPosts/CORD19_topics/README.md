# CORD-19 Topic Modeling

Author: Mark Graves [github.com/markgraves](http://github.com/markgraves)

This folder contains code that builds a topic model for CORD-19 data (version 5 of the data from March 27, 2020)

The corresponding blog post is [Organizing the Emerging COVID-19 Literature](https://roamanalytics.com/covid19-topics).

The folder also contains more recent code (after the blog was posted) to build a topic model for CORD-19 data version 7 (April 10, 2020). See [CHANGELOG](CHANGELOG.md).

## Instructions

To replicate, run `scripts/2020-04-10-v7/fetch_raw_data.sh` (from the `cord19-2020-04-10-v7/` directory) and then run the notebooks in the order listed (within the `notebooks` directory).

## Files inventory

In `cord19-2020-04-10-v7/`
- `scripts/2020-04-10-v7/fetch_raw_data.sh`: shell script to download the raw data (v7)
- `notebooks/2020-04-10-covid19-data-preprocess-clean.ipynb`: notebook to merge and lightly clean the raw data
- `notebooks/2020-04-10-covid19-text-parse-embed-scispacy.ipynb`: notebook to parse the abstracts using scispacy
- `notebooks/2020-04-10-covid19-topics-gensim-mallet-scispacy.ipynb`: notebook to create a topic model using mallet
- `notebooks/*.py` equivalent python files of the corresponding notebook
- `results/2020-04-10-v7/html/*.html` Result of the topic model as web pages (as described in blog post for v5)
- `results/2020-04-10-v7/*.xlsx` Excel files corresponding to the html results

In `cord19-2020-03-27-v5/`
- `scripts/2020-03-27-v5/fetch_raw_data.sh`: shell script to download the raw data (v5)
- `notebooks/2020-03-27-covid19-data-preprocess-clean.ipynb`: notebook to merge and lightly clean the raw data
- `notebooks/2020-03-27-covid19-text-parse-embed-scispacy.ipynb`: notebook to parse the abstracts using scispacy
- `notebooks/2020-03-27-covid19-topics-gensim-mallet-scispacy.ipynb`: notebook to create a topic model using mallet
- `notebooks/*.py` equivalent python files of the corresponding notebook
- `results/*.xlsx` Excel files corresponding to the html in the blog post

In the root directory
- `ldavis.v1.0.0-roam.js` a JavaScript file accessed by the visualization tool

## Questions?

Contact us at <research@roaminsight.com>.