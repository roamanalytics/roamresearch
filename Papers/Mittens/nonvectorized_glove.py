"""nonvectorized_glove.py

This implementation is based in part on Grady Simons'

https://github.com/GradySimon/tensorflow-glove

where credit is also given to Jon Gauthier

https://github.com/hans/glove.py

For the sake of our speed tests, we removed a lot of the functionality
in Simons' version, changed the initialization scheme to match our
own, and made some changes to the way batches were handled. Only our
version is tested by the speed tests in our paper, and any mistakes
are our own!
"""
from __future__ import division
from itertools import product
import numpy as np
import os
from random import shuffle
import tensorflow as tf
import sys

__author__ = 'Nick Dingwall and Christopher Potts'


class GloVeModel:
    def __init__(self, n=100, alpha=3/4, xmax=100, eta=0.05, max_iter=100,
                 batch_size=int(1e6), random_seed=None):
        self.embedding_size = n
        self.scaling_factor = alpha
        self.cooccurrence_cap = xmax
        self.learning_rate = eta
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.batch_size = batch_size

    def _build_graph(self):
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            tf.set_random_seed(self.random_seed)

            count_max = tf.constant(
                [self.cooccurrence_cap], dtype=tf.float32,
                name='max_cooccurrence_count')
            scaling_factor = tf.constant(
                [self.scaling_factor], dtype=tf.float32,
                name="scaling_factor")

            self.__focal_input = tf.placeholder(
                tf.int32, shape=[None], name="focal_words")
            self.__context_input = tf.placeholder(
                tf.int32, shape=[None], name="context_words")
            self.cooccurrence_count = tf.placeholder(
                tf.float32, shape=[None], name="cooccurrence_count")

            focal_embeddings = self._weight_init(
                self.vocab_size, self.embedding_size, "focal_embeddings")
            context_embeddings = self._weight_init(
                self.vocab_size, self.embedding_size, "context_embeddings")

            focal_biases = self._bias_init(
                self.vocab_size, name='focal_biases')
            context_biases = self._bias_init(
                self.vocab_size, name='context_biases')

            focal_embedding = tf.nn.embedding_lookup(
                [focal_embeddings], self.__focal_input)
            context_embedding = tf.nn.embedding_lookup(
                [context_embeddings], self.__context_input)
            focal_bias = tf.nn.embedding_lookup(
                [focal_biases], self.__focal_input)
            context_bias = tf.nn.embedding_lookup(
                [context_biases], self.__context_input)

            weighting_factor = tf.minimum(
                1.0,
                tf.pow(
                    tf.div(self.cooccurrence_count, count_max),
                    scaling_factor))

            embedding_product = tf.reduce_sum(
                tf.multiply(focal_embedding, context_embedding), 1)

            log_cooccurrences = tf.log(tf.to_float(self.cooccurrence_count))

            distance_expr = tf.square(tf.add_n([
                embedding_product,
                focal_bias,
                context_bias,
                tf.negative(log_cooccurrences)]))

            single_losses = tf.multiply(weighting_factor, distance_expr)
            self.__total_loss = tf.reduce_sum(single_losses)
            self.__optimizer = tf.train.AdagradOptimizer(
                self.learning_rate).minimize(self.__total_loss)
            self.__combined_embeddings = tf.add(
                focal_embeddings, context_embeddings,
                name="combined_embeddings")

    def fit(self, X):
        indices = list(range(X.shape[0]))
        self.vocab_size = X.shape[0]
        self._build_graph()
        with tf.Session(graph=self.__graph) as session:
            tf.global_variables_initializer().run()
            for epoch in range(self.max_iter):
                shuffle(indices)
                iterator = enumerate(self.batched_iterator(X, indices))
                for b_num, (i_indices, j_indices, counts) in iterator:
                    self._progressbar("Iter {}, batch {}".format(epoch, b_num+1))
                    feed_dict = {
                        self.__focal_input: i_indices,
                        self.__context_input: j_indices,
                        self.cooccurrence_count: counts}
                    session.run([self.__optimizer], feed_dict=feed_dict)
            return self.__combined_embeddings.eval()

    def batched_iterator(self, X, indices):
        batch = []
        for i, j in product(indices, repeat=2):
            if X[i, j] > 0:
                batch.append((i, j, X[i, j]))
            if len(batch) == self.batch_size:
                yield zip(*batch)
                batch = []
        yield zip(*batch)

    def _weight_init(self, m, n, name):
        """Uses the Xavier Glorot method for initializing weights."""
        x = np.sqrt(6.0/(m+n))
        with tf.name_scope(name) as scope:
            return tf.Variable(
                tf.random_uniform(
                    [m, n], minval=-x, maxval=x), name=name)

    def _bias_init(self, m, name):
         with tf.name_scope(name) as scope:
            return tf.Variable(
                tf.random_uniform(
                    [m], minval=-0.5, maxval=0.5), name=name)

    def _progressbar(self, msg):
        sys.stderr.write('\r')
        sys.stderr.write(msg)
        sys.stderr.flush()


if __name__ == '__main__':

    from utils import correlation_test

    X = np.array([
        [10.0,  0.0,  3.0,  4.0],
        [ 0.0, 10.0,  4.0,  1.0],
        [ 3.0,  4.0, 10.0,  0.0],
        [ 4.0,  1.0,  0.0, 10.0]])

    model = GloVeModel(n=4, max_iter=1000, random_seed=42)
    G = model.fit(X)
    rho = correlation_test(X, G)
    print("Correlation between GloVe dot products and "
          "co-occurence probabilities: {:0.02f}".format(rho['prob']))
