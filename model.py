import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.rnn_cell_impl import RNNCell

def gen_non_linearity(A, non_linearity):
    if non_linearity == "tanh":
        return math_ops.tanh(A)
    elif non_linearity == "sigmoid":
        return math_ops.sigmoid(A)
    elif non_linearity == "relu":
        return gen_math_ops.maximum(A, 0.0)
    elif non_linearity == "quantTanh":
        return gen_math_ops.maximum(gen_math_ops.minimum(A, 1.0), -1.0)
    elif non_linearity == "quantSigm":
        A = (A + 1.0) / 2.0
        return gen_math_ops.maximum(gen_math_ops.minimum(A, 1.0), 0.0)
    elif non_linearity == "quantSigm4":
        A = (A + 2.0) / 4.0
        return gen_math_ops.maximum(gen_math_ops.minimum(A, 1.0), 0.0)
    else:
        # non_linearity is a user specified function
        if not callable(non_linearity):
            raise ValueError("non_linearity is either a callable or a value " +
                             + "['tanh', 'sigmoid', 'relu', 'quantTanh', " +
                             "'quantSigm'")
        return non_linearity(A)

class FastGRNNCell(RNNCell):

    def __init__(self, hidden_size, gate_non_linearity="sigmoid",
                 update_non_linearity="tanh", wRank=None, uRank=None,
                 zetaInit=1.0, nuInit=-4.0, name="FastGRNN", reuse=None):
        super(FastGRNNCell, self).__init__(_reuse=reuse)
        self._hidden_size = hidden_size
        self._gate_non_linearity = gate_non_linearity
        self._update_non_linearity = update_non_linearity
        self._num_weight_matrices = [1, 1]
        self._wRank = wRank
        self._uRank = uRank
        self._zetaInit = zetaInit
        self._nuInit = nuInit
        if wRank is not None:
            self._num_weight_matrices[0] += 1
        if uRank is not None:
            self._num_weight_matrices[1] += 1
        self._name = name
        self._reuse = reuse

    @property
    def state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def gate_non_linearity(self):
        return self._gate_non_linearity

    @property
    def update_non_linearity(self):
        return self._update_non_linearity

    @property
    def wRank(self):
        return self._wRank

    @property
    def uRank(self):
        return self._uRank

    @property
    def num_weight_matrices(self):
        return self._num_weight_matrices

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "FastGRNN"

    def call(self, inputs, state):
        with vs.variable_scope(self._name + "/FastGRNNcell", reuse=self._reuse):

            if self._wRank is None:
                W_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W = vs.get_variable(
                    "W", [inputs.get_shape()[-1], self._hidden_size],
                    initializer=W_matrix_init)
                wComp = math_ops.matmul(inputs, self.W)
            else:
                W_matrix_1_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W1 = vs.get_variable(
                    "W1", [inputs.get_shape()[-1], self._wRank],
                    initializer=W_matrix_1_init)
                W_matrix_2_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W2 = vs.get_variable(
                    "W2", [self._wRank, self._hidden_size],
                    initializer=W_matrix_2_init)
                wComp = math_ops.matmul(
                    math_ops.matmul(inputs, self.W1), self.W2)

            if self._uRank is None:
                U_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U = vs.get_variable(
                    "U", [self._hidden_size, self._hidden_size],
                    initializer=U_matrix_init)
                uComp = math_ops.matmul(state, self.U)
            else:
                U_matrix_1_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U1 = vs.get_variable(
                    "U1", [self._hidden_size, self._uRank],
                    initializer=U_matrix_1_init)
                U_matrix_2_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U2 = vs.get_variable(
                    "U2", [self._uRank, self._hidden_size],
                    initializer=U_matrix_2_init)
                uComp = math_ops.matmul(
                    math_ops.matmul(state, self.U1), self.U2)
            # Init zeta to 6.0 and nu to -6.0 if this doesn't give good
            # results. The inits are hyper-params.
            zeta_init = init_ops.constant_initializer(
                self._zetaInit, dtype=tf.float32)
            self.zeta = vs.get_variable("zeta", [1, 1], initializer=zeta_init)

            nu_init = init_ops.constant_initializer(
                self._nuInit, dtype=tf.float32)
            self.nu = vs.get_variable("nu", [1, 1], initializer=nu_init)

            pre_comp = wComp + uComp

            bias_gate_init = init_ops.constant_initializer(
                1.0, dtype=tf.float32)
            self.bias_gate = vs.get_variable(
                "B_g", [1, self._hidden_size], initializer=bias_gate_init)
            z = gen_non_linearity(pre_comp + self.bias_gate,
                                  self._gate_non_linearity)

            bias_update_init = init_ops.constant_initializer(
                1.0, dtype=tf.float32)
            self.bias_update = vs.get_variable(
                "B_h", [1, self._hidden_size], initializer=bias_update_init)
            c = gen_non_linearity(
                pre_comp + self.bias_update, self._update_non_linearity)
            new_h = z * state + (math_ops.sigmoid(self.zeta) * (1.0 - z) +
                                 math_ops.sigmoid(self.nu)) * c
        return new_h, new_h

    def getVars(self):
        Vars = []
        if self._num_weight_matrices[0] == 1:
            Vars.append(self.W)
        else:
            Vars.extend([self.W1, self.W2])

        if self._num_weight_matrices[1] == 1:
            Vars.append(self.U)
        else:
            Vars.extend([self.U1, self.U2])

        Vars.extend([self.bias_gate, self.bias_update])
        Vars.extend([self.zeta, self.nu])

        return Vars

class Encoder(tf.keras.Model):
  def __init__(self, source_words_count):
    super(Encoder, self).__init__()
    self.emb = tf.keras.layers.Embedding(source_words_count, 64, input_length=4)
    self.fast_grnn = tf.keras.layers.RNN(FastGRNNCell(128, reuse=tf.compat.v1.AUTO_REUSE), return_sequences=True, return_state=True)

  def call(self, x, training=False, mask=None):
    x = self.emb(x)
    output, state = self.fast_grnn(x)
    return output, state

class Decoder(tf.keras.Model):
  def __init__(self, target_words_count):
    super(Decoder, self).__init__()
    self.emb = tf.keras.layers.Embedding(target_words_count, 64, input_length=4)
    self.fast_grnn = tf.keras.layers.RNN(FastGRNNCell(128, reuse=tf.compat.v1.AUTO_REUSE), return_sequences=True, return_state=True)
    self.att = tf.keras.layers.Attention()
    self.dense = tf.keras.layers.Dense(target_words_count, activation='softmax')

  def call(self, inputs, training=False, mask=None):
    x, encoder_output, encoder_hidden = inputs
    x = self.emb(x)

    S, hidden = self.fast_grnn(x, initial_state=[encoder_hidden])
    S_ = tf.concat([encoder_hidden[:, tf.newaxis, :], S[:, :-1, :]], axis=1)
    A = self.att([S_, encoder_output])

    y = tf.concat([S, A], axis=-1)
    return self.dense(y), hidden

class Seq2seq(tf.keras.Model):
  def __init__(self, source_words_count, target_words_count, sos, eos, training=True):
    super(Seq2seq, self).__init__()
    self.enc = Encoder(source_words_count)
    self.dec = Decoder(target_words_count)
    self.sos = sos
    self.eos = eos

  def call(self, inputs, training=False, mask=None):
    if training is True:
      x, y = inputs
      output, hidden = self.enc(x)
      y, _ = self.dec((y, output, hidden))
      return y

    else:
      x = inputs
      H, hidden = self.enc(x)

      y = tf.convert_to_tensor(self.sos)
      y = tf.reshape(y, (1, 1))

      seq = tf.TensorArray(tf.int32, 32)

      for idx in tf.range(32):
        y, hidden = self.dec([y, H, hidden])
        y = tf.cast(tf.argmax(y, axis=-1), dtype=tf.int32)
        y = tf.reshape(y, (1, 1))
        seq = seq.write(idx, y)

        if y == self.eos:
          break

      return tf.reshape(seq.stack(), (1, 32))

@tf.function
def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_accuracy):
  output_labels = labels[:, 1:]
  shifted_labels = labels[:, :-1]
  with tf.GradientTape() as tape:
    predictions = model([inputs, shifted_labels], training=True)
    loss = loss_object(output_labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)

  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)
  train_accuracy(output_labels, predictions)

@tf.function
def test_step(model, inputs):
  return model(inputs, training=False)

if __name__ == "__main__":

  sample = tf.random.uniform(shape=[1, 84], minval=0, maxval=10, dtype=tf.int32)

  encoder = Encoder(10)
  sample_encoder_output, sample_encoder_hidden = encoder(sample)

  # print(sample_encoder_output) (1, 84, 256)
  # print(sample_encoder_hidden) (1, 256)

  decoder = Decoder(10)
  sample_decoder_output, sample_decoder_hidden = decoder((sample, sample_encoder_output, sample_encoder_hidden))

  # print(sample_decoder_output) (1, 84, 10)