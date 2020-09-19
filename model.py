import tensorflow as tf

class Encoder(tf.keras.Model):
  def __init__(self, source_words_count):
    super(Encoder, self).__init__()
    self.emb = tf.keras.layers.Embedding(source_words_count, 64)
    self.gru = tf.keras.layers.GRU(256, return_sequences=True, return_state=True)

  def call(self, x, training=False, mask=None):
    x = self.emb(x)
    output, state = self.gru(x)
    return output, state

class Decoder(tf.keras.Model):
  def __init__(self, target_words_count):
    super(Decoder, self).__init__()
    self.emb = tf.keras.layers.Embedding(target_words_count, 64)
    self.gru = tf.keras.layers.GRU(256, return_sequences=True, return_state=True)
    self.att = tf.keras.layers.Attention()
    self.dense = tf.keras.layers.Dense(target_words_count, activation='softmax')

  def call(self, inputs, training=False, mask=None):
    x, encoder_output, encoder_hidden = inputs
    x = self.emb(x)

    S, hidden = self.gru(x, initial_state=[encoder_hidden])
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

      seq = tf.TensorArray(tf.int32, 64)

      for idx in tf.range(64):
        y, hidden = self.dec([y, H, hidden])
        y = tf.cast(tf.argmax(y, axis=-1), dtype=tf.int32)
        y = tf.reshape(y, (1, 1))
        seq = seq.write(idx, y)

        if y == self.eos:
          break

      return tf.reshape(seq.stack(), (1, 64))

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