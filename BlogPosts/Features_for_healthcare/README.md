# Roam Drug-Disease Dataset

Version: 1.0

Contact: brunogodefroy@roaminsight.com, cgpotts@roaminsight.com


## Background

The dataset consists of sentences extracted from FDA drug labels, with annotations identifying the treatment relation between the drugs described and disease mentions.

About 9.5K sentences have been annotated via crowdsourcing and are expected to be used for training. 500 sentences have been annotated by a team of experts, for evaluation. The inter-annotator agreement between the labels inferred from the crowd and those from the experts is 0.82.

If you use this dataset, please cite this paper:

Yifeng Tao, Bruno Godefroy, Guillaume Genthial, and Christopher Potts. 2018.
Effective Feature Representation for Clinical Text Concept Extraction.
https://arxiv.org/abs/1811.00070

```
@unpublished{roam:clinicalfeatures,
  Author = {Tao, Yifeng and
            Godefroy, Bruno and
            Genthial, Guillaume and
            Potts, Christopher},
  Title = {Effective Feature Representation for Clinical Text
           Concept Extraction},
  Note = {arXiv 1811.00070},
  Year = {2018}}
```

## License

This dataset is licensed under a Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by-sa/4.0).  By taking the action of downloading this dataset, you hereby agree to the associated license terms.


## Annotations

Annotations consist of a label assigned to a group of contiguous tokens (words and punctuation marks).  There is only one possible label per group of token.

Labels:
- 'PREVENTS': the drug stops the affliction from happening or decreases its probability.
- 'TREATS': the drug could cure or participate to cure the affliction or its symptoms.
- 'CONTRAINDICATED_FOR': the drug should not be used because it could worsen the patient's condition.
- 'UNRELATED': other mentions of afflictions


## JSON schema

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

All 'sentence_id' and 'ann_id' are unique in the dataset.  For each annotation, 'text' contains the characters of the sentences from the 'start' index (included) to the 'end' index (excluded).


## Data source

The sentences are drawn from the official FDA drug labels text, a public source of information about how drugs relate to disease states (among other things).

Steps for data preparation:

1. Download drug labels text, section "indications and usage"
2. Tokenize text into sentences
3. Remove special characters, list formatting, references, notes, and line breaks.
4. Remove duplicate sentences
5. Filter out very short sentences (less than 20 characters)
6. Sample 10K of these sentences
