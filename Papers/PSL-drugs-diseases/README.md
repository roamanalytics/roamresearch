# Roam Drug-Disease Relations Datasets

Bruno Godefroy

## Overview

This corpus distribution consists of two files:

1. `drug_disease_relations-graph_examples.jsonl`: The crowdsourced annotations used in the graphs described in the paper.
2. `drug_disease_relations-crf_train.json`: A separate expert-annotated dataset used for training a separate CRF.

Released under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0).


## Citation

```
@unpublished{Godefroy:Potts:2019,
  Author = {Godefroy, Bruno  and  Potts, Christopher},
  Note = {Ms., Roam Analytics and Stanford University. arXiv:1904.00313},
  Title = {Modeling Drug--Disease Relations with Linguistic and Knowledge Graph Constraints},
  Year = {2019}}
```

## drug_disease_relations-graph_examples.jsonl

This dataset contains 8,612 drug-disease relations, labelled using crowdsourcing. Relations are drawn from a subset of 2,434 drug label texts, representative of all the drugs labels released by the FDA. For each drug-disease relation, we collected judgments from 5 independent contributors. All the individual judgments are provided as well as a unique label for each drug-disease relation, inferred from the judgments using Expectation Maximization.

### Relations

For each drug-disease pair, we asked crowdworkers to select one of these labels:

- 'PREVENTS': the drug described prevents the disease.
- 'TREATS': the drug described treats the disease.
- 'TREATS_OUTCOMES': the drug described treats an affliction which is an outcome of the disease.
- 'NOT_ESTABLISHED': the safety/effectiveness of the drug described has not been established for the disease.
- 'NOT_RECOMMENDED': the drug is not recommended for the disease.
- 'OTHER': any other relation.

### JSON schema

The dataset is provided under the JSON Lines format.  Each line is a valid JSON value and refers to a particular drug.  This is, for example, the JSON value for methoxsalen:

```
{
  "RXCUIs": [197947, 207073, 207074],
  "text": "INDICATIONS AND USAGE Methoxsalen is indicated for ...",
  "disease_mentions": {
    "psoriasis": {
       "ICD10s": ["L40", "L403", "L409"],
       "text_offsets": [[159, 168]],
       "relation": "TREATS",
       "agreement": 0.8,
       "judgments": [
         {"worker_id": 2, "response": "TREATS"},
         {"worker_id": 4, "response": "TREATS"},
         {"worker_id": 11, "response": "PREVENTS"},
         {"worker_id": 80, "response": "TREATS"},
         {"worker_id": 81, "response": "TREATS"}
       ]
    },
    ...
  }
}
```

Additional details on the fields:

 - `RXCUIs`: a list of RXCUI codes identifying the drug.
 - `text`: content of the "indications and usage" section in the drug label.
 - `disease_mentions`: the disease mentions matched in the text.
 - `ICD10s`: a list of ICD10 codes identifying the disease.
 - `text_offsets`: the locations of mentions of the disease in text; for each location, the first value is the start index (included) and the second values is the end index (excluded).
 - `relation`: the drug-disease relation inferred from the judgments collected.
 - `agreement`: the agreement between the judgments and the inferred label.
 - `judgments`: the 5 judgments collected via crowdsourcing.
 - `worker_id`: an identifier of the worker who provided this judgment.
 - `response`: the response provided by the worker.

### Drug labels

We have labelled relations for diseases mentioned in a subset of the official FDA drug labels, a public source of information about how drugs relate to disease states (among other things).

We considered jointly products that refer to the same drug (there is usually multiple versions or brands for the same medication). We have also merged products when their names or some of their drug labels were matching exactly. Then, for each drug, we have considered only the most recent drug label, and, for drug labels released at the same date, the one containing the most characters (assuming it contains more information).


### Matching disease mentions

Disease mentions in text have been matched with a lexicon. This lexicon is derived from historical ICD-9 and ICD-10 code sets, SNOMED-CT, the Disease Ontology, and the Wikidata graph.


### Judgment aggregation

This step consists in inferring a unique label for each drug-disease relation, by aggregating the judgments collected. The result is stored in the 'relation' field.

For each drug-disease pair, this process was performed in 2 steps:

1. Decide whether the drug is approved for the disease, ie. whether the true label is in the set of labels ('TREATS', 'TREATS_OUTCOMES', 'PREVENTS') or in
the set of labels ('OTHER', 'NOT_ESTABLISHED', 'NOT_RECOMMENDED').
2. Decide which label is the most appropriate in the set of labels selected at  the previous step.

For step 1, we applied Expectation Maximization (EM), essentially as in Dawid and Skene (1979). The algorithm estimates the reliability of each worker and use it to weight their contributions.  For step 2, we chose the label, in the selected set, on which the most judgments agree. In case of equality, we used the following order of priority:

- 'TREATS' > 'TREATS_OUTCOMES' > 'PREVENTS'
- 'OTHER' > 'NOT_ESTABLISHED' > 'NOT_RECOMMENDED'

Agreement after step 1:

Relations      | Agreement
-------------- | -------------
'approved'     | 0.906
'not approved' | 0.454

Overall agreement:

Relations         | Agreement
-------------     | -------------
'PREVENTS'        | 0.630
'TREATS'          | 0.673
'TREATS_OUTCOMES' | 0.671
'NOT_ESTABLISHED' | 0.351
'NOT_RECOMMENDED' | 0.495
'OTHER'           | 0.359


### Statistics

- Drug-disease pairs: 8,612.
- Drug labels: 2,434 (related to 13,673 distinct RXCUI codes).
- Disease mentions: 12,586 (1,356 distinct diseases).
- Collected judgments: 43,060, from 431 contributors in 14 countries.

Distribution of the aggregated labels:

Relations         | Count
-------------     | -------------
'PREVENTS'        | 154
'TREATS'          | 4,425
'TREATS_OUTCOMES' | 2,268
'NOT_ESTABLISHED' | 241
'NOT_RECOMMENDED' | 262
'OTHER'           | 1,262



## drug_disease_relations-crf_train.json

The dataset contains 2K sentences drawn from FDA drug labels, with expert annotations identifying the treatment relation between the drugs described and disease mentions.

### Annotations

Annotations consist of a label assigned to a group of contiguous tokens (words and punctuation marks). There is only one possible label per group of token.

Labels:

- 'PREVENTS': the drug stops the affliction from happening or decreases its probability.
- 'TREATS': the drug could cure or participate to cure the affliction.
- 'TREATS_OUTCOMES': the drug could cure or participate to cure outcomes of the affliction.
- 'NOT_RECOMMENDED': the drug should not be used because it could worsen the patient's condition.


### JSON schema

Annotations are grouped by sentence. For each sentence, we have the following fields:

```
{
  'sentence': str,
  'sentence_id': str,
  'annotations': [
    {
      'ann_id': str
      'text': str,
      'start': int,
      'end': int,
      'label': str
    },
    ...
  ]
}
```

All `sentence_id` and `ann_id` values are unique in the dataset.  For each annotation, `text` contains the characters of the sentences from the `start` index (included) to the `end` index (excluded).


### Data source

The sentences are drawn from the official FDA drug labels text, a public source of information about how drugs relate to disease states (among other things).

Steps for data preparation:

1. Download drug labels text, section "indications and usage"
2. Tokenize text into sentences
3. Remove special characters, list formatting, references, notes, and line breaks.
4. Remove duplicate sentences
5. Filter out very short sentences (less than 20 characters)
6. Sample 2K of these sentences
