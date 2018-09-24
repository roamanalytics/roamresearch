# Modern Tensorflow for NLP

Author: Guillaume Genthial

Date: 2018-09-24


__TL;DR__: Recent changes to TensorFlow make it much easier to develop and productionize deep learning models. We've created a notebook full of code snippets that illustrate the new design patterns and features, including a number of tips and tricks specific to developing NLP models.

* [The notebook of code snippets](modern-tensorflow.ipynb) [[nbviewer view for better formatting](http://nbviewer.jupyter.org/github/roamanalytics/roamresearch/blob/master/BlogPosts/Modern_TensorFlow/modern-tensorflow.ipynb)]
* [Blog post version of this README](https://roamanalytics.com/good-practices-in-modern-tensorflow-for-nlp/)


## The evolution of deep learning libraries

Another year, another contest in the small world of deep learning libraries. In 2016, rising from the ashes of [Theano](http://deeplearning.net/software/theano/) (Montreal University's precursor), Google's [TensorFlow](https://www.tensorflow.org/) quickly imposed itself as the industry leader, gaining popularity for its 'ease of use'.

Today, two years later, a new challenger threatens TensorFlow's hegemony. Born at Facebook, [PyTorch](https://pytorch.org/) is quickly gaining popularity, especially among researchers. If you hang out with CS grad students, chances are that one of them will tell you how they 're-implemented an entire model in PyTorch overnight'.

If you're a firm believer in the positive impacts of (fair) competition, it won't come as a surprise, but this led to major welcome changes and additions to TensorFlow, starting especially at [version 1.9](https://github.com/tensorflow/tensorflow/releases/tag/v1.9.0).

At Roam, we carefully reviewed those additions from an NLP-oriented perspective (though most of it applies to other areas as well). We compiled [a short guide](modern-tensorflow.ipynb) to help engineers quickly bootstrap their deep learning projects. We are making it publicly available in the hope that it will help other engineers and scientists, from academia and industry.

## This notebook

Our [notebook of code snippets](modern-tensorflow.ipynb) covers three main topics:

1. Eager execution (PyTorch-like): yes, you can now print actual values of the graph.
2. `tf.data`: a better, faster and stronger way of inputing data to the graph (thanks to Derek Murray et al!)
3. `tf.estimator`: a new high-level API for model-building.

For (possibly former) users of TensorFlow who went through the concepts of `tf.Session`, `tf.placeholder`, etc. and had to write their own `Model` classes: this belongs to the past. You can now have a state-of-the-art deep learning model up and running with just a few lines of code. Less boilerplate code, better readability.

There are still major differences between PyTorch and TensorFlow, and each has its strengths and weaknesses. This notebook will walk you through some of modern TensorFlow's strengths. For instance, `tf.estimator` gives you a lot of things for 'free', including TensorBoard, model serialization, RESTful API serving of your model, and a unified interface that makes reusing others' code easier.

Because NLP has its own needs (vocabularies, tokenization, padding, etc.), we also provide some tips and tricks for using TensorFlow in the right way with text, and try to find the right tradeoff between efficiency and ease-of-implementation. (We prefer to use `tf.data.Dataset.from_generator` rather than `tf.data.TextLineDataset`, and we also like the `tf.contrib.lookup` module for vocabularies, etc.)

The main takeaway of our review is that the latest versions of TensorFlow addressed most of the concerns its users had, making it much more friendly, flexible, and efficient. We found that it successfully reduces the need for high-level wrappers like Keras by allowing fast prototyping that almost immediately converts into productionized models &ndash; this, for a startup, is a must-have!

## What lies ahead?

It's hard to predict the future, but some of the battles that lie ahead probably involve cloud computing (will Google's [TPU](https://en.wikipedia.org/wiki/Tensor_processing_unit) force most companies to use TensorFlow?), or weights exchange ([ONNX](https://onnx.ai/)?) and framework compatibility (at least of serialized models). In the meantime, let's try to use the existing tools at their best!
