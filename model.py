import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import *

"""
reference: https://heung-bae-lee.github.io/2020/01/22/deep_learning_11/
"""

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.lstm = LSTM(512, return_state=True)

    def call(self, x, training=False, mask=None):
        H, h, c = self.lstm(x)
        return H, h, c

class Decoder(tf.keras.Model):
    def __init__(self, vocab_length):
        super(Decoder, self).__init__()
        self.vocab_length = vocab_length
        self.lstm = LSTM(512, return_state=True, return_sequences=True)
        self.att = Attention()
        self.dense = Dense(self.vocab_length, activation='softmax')

    def call(self, inputs, training=False, mask=None):
        x, h, c, H = inputs
        s, h, c = self.lstm(x, initial_state=[h, c])
        S_ = tf.concat([h[:, tf.newaxis, :], s[:, :-1, :]], axis=1)
        A = self.att([S_, H])
        y = tf.concat([s, A], axis=-1)

        return self.dense(y), h, c
        
class Seq2Seq(tf.keras.Model):
    def __init__(self, sequence_length, embedding_size, vocab_length):
        super(Seq2Seq, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.enc = Encoder()
        self.dec = Decoder(vocab_length = vocab_length)

    def call(self, inputs, training=False, mask=None):
        if training is True:
            x, y = inputs
            H, h, c = self.enc(x)
            y, _, _ = self.dec((y, h, c, H))
            return y
        else:
            # 추론 중에는 디코더에 패딩으로 채운 값을 입력으로 줘야한다.
            x = inputs
            H, h, c = self.enc(x)

            y = tf.convert_to_tensor(np.zeros((1, self.sequence_length, self.embedding_size)), dtype=tf.float32)

            y, h, c = self.dec((y, h, c, H))
            y = tf.cast(tf.argmax(y, axis=-1), dtype=tf.int32)

            return tf.reshape(y, (1, self.sequence_length))

@tf.function
def train_step(model, encoder_input, decoder_input, decoder_output, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model([encoder_input, decoder_input], training=True)
        loss = loss_object(decoder_output, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(decoder_output, predictions)

@tf.function
def test_step(model, inputs):
    return model(inputs, training=False)