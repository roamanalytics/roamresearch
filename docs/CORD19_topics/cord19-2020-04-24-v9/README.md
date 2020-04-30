# COVID-19 Topic Modeling

This folder contains topic modeling results for [CORD-19 data](https://pages.semanticscholar.org/coronavirus-research) (version 9 of the data from April 26, 2020).

It extends the results described in the blog post [Organizing the Emerging COVID-19 Literature](https://roamanalytics.com/covid19-topics) (on v5 of the data).

The web pages consist of:
- A 75-topic model on 43K abstracts ([overview](text-ents-en-75/index.html)).
- The topic model can be viewed as a [collection of relevant terms](text-ents-en-75/relevant_terms.html) for each topic and through [topic visualization](text-ents-en-75/pyldavis_text-ents-en_75.html).
- For each topic, the documents predominantly on that topic are listed, with link out to full article where available.
- Trends over time for each topic are shown as a heatmap (yearly prior to Jan 2020, and weekly afterwards).
- Each of the topics is also modeled as [sub-topics](text-ents-en-75/subtopics.html). 

Notebooks and code to generate the model are at [https://github.com/roamanalytics/roamresearch/tree/master/BlogPosts/CORD19_topics/](https://github.com/roamanalytics/roamresearch/tree/master/BlogPosts/CORD19_topics/)

## Questions?

Contact us at <research@roaminsight.com>.

Disclaimer: These are preliminary results from works-in-progress based upon data collected by others which we are releasing in the hope they may help assist in on-going, public, collaborative efforts to organize and explore the rapidly emerging COVID-19 literature. We make no claim of validity or that it is fit for any purpose whatsoever. By design it will exclude potentially relevant articles.
