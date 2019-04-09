import numpy as np
import tensorflow as tf
from utils import evaluate_rnn

__author__ = "Chris Potts and Yifeng Tao"


class model:
  """ This is the class for four models, which is specified by the `model_name`.
  Fit a CRF labeler for sequences.

  Parameters
  ----------
  vocab: list
      Tokenized words in the dataset.
  model_name: str
      Can only be one of the following four:
          "rand_lstm_crf": rand-LSTM-CRF
          "elmo_lstm_crf": ELMo-LSTM-CRF
          "hb_crf":        HB-CRF
          "combined":      ELMo-LSTM-CRF-HB
  max_iter: int
      Number of samples trained through.
  eta: float
      Step size.
  batch_size: int
      Batch size for training.
  test_batch_size: int
      Batch size for testing.
  hid_dim: int
      Dimension of hidden layer after the LSTM.
      Only take effect when `model_name` is "elmo_lstm_crf" or "combined".
  hid_hb_dim: int
      Dimension of hidden layer after hand-built feature.
      Only takes effect when `model_name` is "combined".
  emb_dim: int
      Dimension of embedding layer.
      Only takes effect when `model_name` is `elmo_lstm_crf` or `combined`.
  feat_dim: int
      Dimension of hand-built feature.
      Only takes effect when `model_name` is `hb_crf` or `combined`.
  max_len: int
      Max length of sentence in a batch of sentences. Will change when
      calling functions `self.get_minibatch_train()` and
      `self.get_minibatch_test()`.
  hid_act: tf.nn activation
  feat2idx: dict
      Feature name to feature index dictionary for hand-built features.
  c1: float
      Coefficient of L1-regularizer.
  c2: float
      Coefficient of L2-regularizer.
  """

  def __init__(
      self,
      vocab=[],
      model_name="combined",
      max_iter=4096,
      eta=0.01,
      batch_size=64,
      test_batch_size=128,
      hid_dim=50,
      hid_hb_dim=100,
      emb_dim=3072,
      feat_dim=85177,
      max_len=374,
      hid_act=tf.nn.tanh,
      feat2idx={},
      c1=0,
      c2=0
      ):
    self.vocab=vocab
    self.model_name=model_name
    self.max_iter=max_iter
    self.eta=eta
    self.batch_size=batch_size
    self.test_batch_size=test_batch_size
    self.hid_dim=hid_dim
    self.hid_hb_dim=hid_hb_dim
    self.emb_dim=emb_dim
    self.feat_dim=feat_dim
    self.max_len=max_len
    self.hid_act=hid_act
    self.feat2idx=feat2idx
    self.c1=c1
    self.c2=c2


  def build_graph(self):
    """ Build the tensorflow graph for one of the following models:
      "rand_lstm_crf": rand-LSTM-CRF
      "elmo_lstm_crf": ELMo-LSTM-CRF
      "hb_crf":        HB-CRF
      "combined":      ELMo-LSTM-CRF-HB
      which is determined by `self.model_name`.

    Cost node is also built in this function.
    """

    # Dropout is used for the dense layer after HB feature in
    # `combined` model. Default value = 0.5.
    self.keep_prob = tf.placeholder(tf.float32)
    # Lengths of sentences
    self.ex_lens = tf.placeholder(tf.int32, [None])

    if self.model_name == "rand_lstm_crf":
      self.embedding_matrix = tf.Variable(
          tf.truncated_normal([len(self.vocab), self.emb_dim]), trainable=True)
      # Each position is an integer index of token: (batch_size, max_len)
      self.X = tf.placeholder(tf.int32, [None, None])
      # (batch_size, max_len, emb_dim)
      self.sent_emb = tf.nn.embedding_lookup(self.embedding_matrix, self.X)

      self.cell = tf.nn.rnn_cell.LSTMCell(self.hid_dim, activation=self.hid_act)
      # Dropout here to prevent overfitting.
      self.cell = tf.nn.rnn_cell.DropoutWrapper(
          self.cell, input_keep_prob=0.5, output_keep_prob=0.5)
      # (batch_size, max_len, hid_dim)
      hiddens, _ = tf.nn.dynamic_rnn(
          self.cell,
          self.sent_emb,
          dtype=tf.float32,
          sequence_length=self.ex_lens)

    # ELMo embedding input for `combined` and `elmo_lstm_crf` models.
    if self.model_name in ["elmo_lstm_crf", "combined"]:
      # (batch_size, max_len, emb_dim)
      self.elmo = tf.placeholder(tf.float32, [None, None, self.emb_dim])
      self.cell = tf.nn.rnn_cell.LSTMCell(self.hid_dim, activation=self.hid_act)
      # No need of dropout to prevent overfitting here.
      # self.cell = tf.nn.rnn_cell.DropoutWrapper(
      #         self.cell, input_keep_prob=0.5, output_keep_prob=0.5)

      # hiddens: (batch_size, max_len, hid_dim)
      hiddens, _ = tf.nn.dynamic_rnn(
          self.cell,
          self.elmo,
          dtype=tf.float32,
          sequence_length=self.ex_lens)

    # HB sparse feature input for `combined` and `hb_crf` models.
    if self.model_name in ["hb_crf", "combined"]:
      # (batch_size, max_len, feat_dim)
      self.feats = tf.placeholder(tf.float32, [None, None, self.feat_dim])

    # Build up the architectures below.
    if self.model_name == "combined":
      # (batch_size, max_len, hid_hb_dim)
      compress = tf.layers.dense(
          inputs=self.feats,
          units=self.hid_hb_dim,
          activation=tf.nn.relu,
          use_bias=True,
          name="W_compress")
      compress = tf.nn.dropout(compress, self.keep_prob)
      # (batch_size, max_len, hid_dim+feat_dim)
      feature = tf.concat( [hiddens, compress], axis=2)
      self.scores = tf.layers.dense(
          inputs=feature,
          units=self.output_dim,
          activation=None,
          use_bias=True,
          name="W_combined")
    elif self.model_name in ["rand_lstm_crf", "elmo_lstm_crf"]:
      self.scores = tf.layers.dense(
          inputs=hiddens,
          units=self.output_dim,
          activation=None,
          use_bias=True,
          name="W_lstm")
    elif self.model_name == "hb_crf":
      self.scores = tf.layers.dense(
          inputs=self.feats,
          units=self.output_dim,
          activation=None,
          use_bias=True,
          name="W_hb")

    # Ground truth label indices: (batch_size, max_len)
    self.tag_indices = tf.placeholder(
        tf.int32, shape=[None, None])

    # CRF log likelihood and param.
    log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
        inputs=self.scores, tag_indices=self.tag_indices, sequence_lengths=self.ex_lens)

    # CRF decode: used for prediction function.
    self.viterbi_sequence, _ = tf.contrib.crf.crf_decode(
        potentials=self.scores, transition_params=self.transition_params, sequence_length=self.ex_lens)

    # L1 and L2 regularizer terms.
    if self.model_name == "combined":
      regularizer = self.c2*tf.nn.l2_loss( tf.trainable_variables("rnn/lstm_cell/kernel")[0] )+\
          self.c2*tf.nn.l2_loss( tf.trainable_variables("W_compress/kernel")[0] )+\
          self.c2*tf.nn.l2_loss( tf.trainable_variables("W_combined/kernel")[0] )+\
          self.c2*tf.nn.l2_loss( tf.trainable_variables("transitions")[0] )+\
          self.c1*tf.reduce_sum(tf.abs( tf.trainable_variables("rnn/lstm_cell/kernel")[0] ))+\
          self.c1*tf.reduce_sum(tf.abs( tf.trainable_variables("W_compress/kernel")[0] ))+\
          self.c1*tf.reduce_sum(tf.abs( tf.trainable_variables("W_combined/kernel")[0] ))+\
          self.c1*tf.reduce_sum(tf.abs( tf.trainable_variables("transitions")[0] ))

    elif self.model_name in ["elmo_lstm_crf", "rand_lstm_crf"]:
      regularizer = self.c2*tf.nn.l2_loss( tf.trainable_variables("rnn/lstm_cell/kernel")[0] )+\
          self.c2*tf.nn.l2_loss( tf.trainable_variables("W_lstm/kernel")[0] )+\
          self.c2*tf.nn.l2_loss( tf.trainable_variables("transitions")[0] )+\
          self.c1*tf.reduce_sum(tf.abs( tf.trainable_variables("rnn/lstm_cell/kernel")[0] ))+\
          self.c1*tf.reduce_sum(tf.abs( tf.trainable_variables("W_lstm/kernel")[0] ))+\
          self.c1*tf.reduce_sum(tf.abs( tf.trainable_variables("transitions")[0] ))
    elif self.model_name == "hb_crf":
      regularizer = self.c2*tf.nn.l2_loss( tf.trainable_variables("W_hb/kernel")[0] )+\
          self.c2*tf.nn.l2_loss( tf.trainable_variables("transitions")[0] )+\
          self.c1*tf.reduce_sum(tf.abs( tf.trainable_variables("W_hb/kernel")[0] ))+\
          self.c1*tf.reduce_sum(tf.abs( tf.trainable_variables("transitions")[0] ))

    # Loss function.
    self.cost = tf.reduce_mean(-log_likelihood)+regularizer


  def get_optimizer(self):
    return tf.train.AdamOptimizer(self.eta).minimize(self.cost)


  def train_dict(self, X, X_elmo, X_sparse, y, ex_lens):
    """ Feed `X_elmo` to the placeholder that defines ELMo embedding input, and
    `X_sparse` to the placeholder that defines HB feature input, and
    `y` to the placeholder that defines the output, and
    `ex_lens` to the placeholder that defines the lengths of examples.

    Default `keep_prob` value is fed in this function.

    This is used during training.
    """

    #keep_prob = 0.5 by default
    feed = {self.ex_lens: ex_lens,
            self.keep_prob: 0.5}
    if self.model_name in ["elmo_lstm_crf", "combined"]:
        feed[self.elmo] = X_elmo
    if self.model_name in ["hb_crf", "combined"]:
        feed[self.feats] = X_sparse
    if self.model_name == "rand_lstm_crf":
        feed[self.X] = X

    # CRF output: (batch_size, max_len)
    # Each position is the index of label.
    new_y = np.zeros((len(y),self.max_len),dtype="int")
    for idx_label_seq, label_seq in enumerate(y):
      for idx_label, label in enumerate(label_seq):
        new_y[idx_label_seq,idx_label] = self.classes.index(label)
    feed[self.tag_indices] = new_y

    return feed


  def test_dict(self, X, X_elmo, X_sparse, ex_lens):
    """ Feed `X_elmo` to the placeholder that defines ELMo embedding input, and
    `X_sparse` to the placeholder that defines HB feature input, and
    `y` to the placeholder that defines the output, and
    `ex_lens` to the placeholder that defines the lengths of examples.

    `keep_prob` is fed with value of 1.0.

    This is used during testing.
    """

    feed = {self.ex_lens: ex_lens,
            self.keep_prob: 1.0}
    if self.model_name in ["combined", "elmo_lstm_crf"]:
      feed[self.elmo] = X_elmo
    if self.model_name in ["combined", "hb_crf"]:
      feed[self.feats] = X_sparse
    if self.model_name == "rand_lstm_crf":
      feed[self.X] = X

    return feed


  def predict(self, X, X_elmo, X_sparse, ex_lens):
    """ Return CRF classifier predictions.

    Parameters
    ----------
    X_elmo: np.array
    X_sparse: np.array
    ex_lens: list of int

    Returns
    -------
    preds: list
    """

    feed = self.test_dict(X, X_elmo, X_sparse, ex_lens)
    viterbi_sequence = self.sess.run(self.viterbi_sequence, feed_dict=feed)
    preds = [ [ self.classes[x] for x in sent[:ex_lens[idx]] ]
      for idx, sent in enumerate(viterbi_sequence)]

    return preds


  def _convert_X(self, X):
    """ Map the raw sentences (tokens) into indices according to the vocabulary.

    Parameters
    ----------
    X: list of list of str
      sentences in the dataset, each sentence consists of a list of tokens/words

    Returns
    -------
    new_X: list of list of int
      sentences in the dataset, each sentence consists of a list of words indices
    """

    new_X = np.zeros((len(X), self.max_len), dtype="int")
    index = dict(zip(self.vocab, range(len(self.vocab))))
    for i in range(new_X.shape[0]):
      vals = [ index[w] for w in X[i] ]
      tmp = np.zeros((self.max_len,), dtype="int")
      tmp[0: len(vals)] = vals
      new_X[i] = tmp

    return new_X


  def _get_minibatch_indices(self, n_sample, batch_size, index, mode):
    """ Return a list of indices of a minibatch training/test samples.
    mode ("train"/"test") determines different returned list of indices.

    Parameters
    ----------
    n_sample: int
      number of samples in the dataset
    batch_size: int
      minibatch size of training or test sets
    index: int
      starting index of a minibatch
    mode: str
      "train": each batch has exactly batch_size samples, can cycle around
      "test": the last batch may have smaller batch sizes, can't cycle around

    Returns
    -------
    list of int
      indices of a minibatch of samples
    """

    assert mode in ["train", "test"]
    if mode == "train":
      return [idx%n_sample for idx in range(index, index+batch_size)]
    elif mode == "test":
      return [idx for idx in range(min(index,n_sample), min(index+batch_size,n_sample))]


  def get_minibatch_data(self, X, X_elmo, X_sparse, y, ex_lens, batch_size, index, mode):
    """ Return a minibatch of training/test dataset with exact size of `batch_size`.
    The returned batch of data will cycle around when index exceeds the
    size of dataset.
    `self.max_len` is changed in the function.

    Parameters
    ----------
    X: list of list of str
    X_elmo: np.array
    X_sparse: np.array
    y: list
    ex_lens: list
    batch_size: int
    index: int

    Returns
    -------
    batch_X: np.array
    batch_X_feature: np.array
    batch_y: list
    batch_ex_lens: list
    """

    indices = self._get_minibatch_indices(len(y), batch_size, index, mode)

    batch_y = [ y[idx] for idx in indices ]
    batch_ex_lens = [ ex_lens[idx] for idx in indices ]
    self.max_len = max(batch_ex_lens)

    if self.model_name in ["elmo_lstm_crf", "combined"]:
      batch_X_elmo = X_elmo[ indices ]
      batch_X_elmo = np.asarray(batch_X_elmo, dtype=np.float32)
      batch_X_elmo = batch_X_elmo[:,0:self.max_len,:]
    else: # ["rand_lstm_crf", "hb_crf"]
        batch_X_elmo = []

    if self.model_name == "rand_lstm_crf":
      batch_X = [ X[idx] for idx in indices ]
      batch_X = self._convert_X(batch_X)
    else: # ["elmo_lstm_crf", "hb_crf", "combined"]
      batch_X = []

    if self.model_name in ["hb_crf", "combined"]:
      batch_X_f = [ X_sparse[idx] for idx in indices ]
      batch_X_sparse = np.zeros(
          (len(batch_X_f), self.max_len, self.feat_dim), dtype="float32")
      for idx_sent, sent in enumerate(batch_X_f):
        for idx_word, word in enumerate(sent):
          for k in word.keys():
            if type(word[k]) is not float:
              f = "str_"+k+"_"+str(word[k])
              batch_X_sparse[idx_sent, idx_word, self.feat2idx[f]] = 1
            else:
              f = "float_"+k
              batch_X_sparse[idx_sent, idx_word, self.feat2idx[f]] = word[k]
    else: # ["rand_lstm_crf", "elmo_lstm_crf"]
      batch_X_sparse = []

    return batch_X, batch_X_elmo, batch_X_sparse, batch_y, batch_ex_lens


  def fit(self, X, X_elmo, X_sparse, y, **kwargs):
    """ Standard `fit` method.

    Parameters
    ----------
    X_elmo: np.array
    X_sparse: np.array
    y: list
    kwargs: dict
        For passing other parameters, e.g., a test set that we want to
        monitor performance on.

    Returns
    -------
    self
    """

    # Output file to record evaluations results.
    log = kwargs.get("log", "default_log.txt")

    # When `iter_train` is multiple of `inc`, we evaluate on test set.
    inc = kwargs.get("inc", 0)

    # Prepare mapping for output.
    self.classes = sorted({c for seq in y for c in seq})
    self.output_dim = len(self.classes)

    y_test = kwargs.get("y_test")
    X_test = kwargs.get("X_test")
    X_elmo_test = kwargs.get("X_elmo_test")
    X_sparse_test = kwargs.get("X_sparse_test")

    ex_lens = [len(yy) for yy in y]
    ex_lens_test = [len(yy) for yy in y_test]

    # Build graph.
    tf.reset_default_graph()
    self.sess = tf.InteractiveSession()
    self.build_graph()
    self.optimizer = self.get_optimizer()
    self.sess.run(tf.global_variables_initializer())

    # Train the model with minibatch over training set cyclely.
    for iter_train in range(0, self.max_iter+1, self.batch_size):
      losses = []

      # Get a minibatch of training data.
      batch_X, batch_X_elmo, batch_X_sparse, batch_y, batch_ex_lens = self.get_minibatch_data(
          X, X_elmo, X_sparse, y, ex_lens, self.batch_size, iter_train, "train")

      _, loss, _ = self.sess.run(
          [self.optimizer, self.cost, self.transition_params],
          feed_dict=self.train_dict(batch_X, batch_X_elmo, batch_X_sparse, batch_y, batch_ex_lens)
          )
      losses.append(loss)

      # Evaluation on whole test set.
      if (iter_train % inc == 0) and iter_train:
        loss = np.mean(losses)
        losses = []

        self.preds = []
        for iter_test in range(0, len(y_test), self.test_batch_size):
          batch_X, batch_X_elmo, batch_X_sparse, batch_y, batch_ex_lens = self.get_minibatch_data(
              X_test, X_elmo_test, X_sparse_test, y_test, ex_lens_test,
              self.test_batch_size, iter_test, "test")
          self.preds = self.preds + self.predict(batch_X, batch_X_elmo, batch_X_sparse, batch_ex_lens)

        # Evaluation metrics.
        self.eval_test = evaluate_rnn(y_test, self.preds)

        strings = "[{},{}], loss:{}, test-f1:{}".format(
            iter_train//len(y),iter_train%len(y),loss,self.eval_test["f1_macro"])
        print(strings)
        with open(log, "a") as f:
          f.write(strings+"\n")

    return self

