import tensorflow as tf

class Encoder(tf.keras.Model):
  def __init__(self, source_words_count):
    super(Encoder, self).__init__()
    # 2000개의 단어들을 64크기의 vector로 Embedding해줌.
    self.emb = tf.keras.layers.Embedding(source_words_count, 128)
    # return_state는 return하는 Output에 최근의 state를 더해주느냐에 대한 옵션
    # 즉, Hidden state와 Cell state를 출력해주기 위한 옵션이라고 볼 수 있다.
    # default는 False이므로 주의하자!
    # return_sequence=True로하는 이유는 Attention mechanism을 사용할 때 우리가 key와 value는
    # Encoder에서 나오는 Hidden state 부분을 사용했어야 했다. 그러므로 모든 Hidden State를 사용하기 위해 바꿔준다.
    self.lstm = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)

  def call(self, x, training=False, mask=None):
    x = self.emb(x)
    H, h, c = self.lstm(x)
    return H, h, c

class Decoder(tf.keras.Model):
  def __init__(self, target_words_count):
    super(Decoder, self).__init__()
    self.emb = tf.keras.layers.Embedding(target_words_count, 128)
    # return_sequence는 return 할 Output을 full sequence 또는 Sequence의 마지막에서 출력할지를 결정하는 옵션
    # False는 마지막에만 출력, True는 모든 곳에서의 출력
    self.lstm = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)
    # LSTM 출력에다가 Attention value를 dense에 넘겨주는 것이 Attention mechanism이므로
    self.att = tf.keras.layers.Attention()
    self.dense = tf.keras.layers.Dense(target_words_count, activation='softmax')

  def call(self, inputs, training=False, mask=None):
    # x : shifted output, s0 : Decoder단의 처음들어오는 Hidden state
    # c0 : Decoder단의 처음들어오는 cell state H: Encoder단의 Hidden state(Key와 value로 사용)
    x, s0, c0, H = inputs
    x = self.emb(x)

    # initial_state는 셀의 첫 번째 호출로 전달 될 초기 상태 텐서 목록을 의미
    # 이전의 Encoder에서 만들어진 Hidden state와 Cell state를 입력으로 받아야 하므로
    # S : Hidden state를 전부다 모아놓은 것이 될 것이다.(Query로 사용)
    S, h, c = self.lstm(x, initial_state=[s0, c0])

    # Query로 사용할 때는 하나 앞선 시점을 사용해줘야 하므로
    # s0가 제일 앞에 입력으로 들어가는데 현재 Encoder 부분에서의 출력이 batch 크기에 따라서 length가 현재 1이기 때문에 2차원형태로 들어오게 된다.
    # 그러므로 이제 3차원 형태로 확장해 주기 위해서 newaxis를 넣어준다.
    # 또한 decoder의 S(Hidden state) 중에 마지막은 예측할 다음이 없으므로 배제해준다.
    S_ = tf.concat([s0[:, tf.newaxis, :], S[:, :-1, :]], axis=1)

    # Attention 적용
    # 아래 []안에는 원래 Query, Key와 value 순으로 입력해야하는데 아래처럼 두가지만 입력한다면
    # 마지막 것을 Key와 value로 사용한다.
    A = self.att([S_, H])

    y = tf.concat([S, A], axis=-1)
    return self.dense(y), h, c

class Seq2seq(tf.keras.Model):
  def __init__(self, source_words_count, target_words_count, sos, eos, training=True):
    super(Seq2seq, self).__init__()
    self.enc = Encoder(source_words_count)
    self.dec = Decoder(target_words_count)
    self.sos = sos
    self.eos = eos

  def call(self, inputs, training=False, mask=None):
    if training is True:
      # 학습을 하기 위해서는 우리가 입력과 출력 두가지를 다 알고 있어야 한다.
      # 출력이 필요한 이유는 Decoder단의 입력으로 shited_ouput을 넣어주게 되어있기 때문이다.
      x, y = inputs

      # LSTM으로 구현되었기 때문에 Hidden State와 Cell State를 출력으로 내준다.
      H, h, c = self.enc(x)

      # Hidden state와 cell state, shifted output을 초기값으로 입력 받고
      # 출력으로 나오는 y는 Decoder의 결과이기 때문에 전체 문장이 될 것이다.
      y, _, _ = self.dec((y, h, c, H))
      return y

    else:
      x = inputs
      H, h, c = self.enc(x)

      # Decoder 단에 제일 먼저 sos를 넣어주게끔 tensor화시키고
      y = tf.convert_to_tensor(self.sos)
      # shape을 맞춰주기 위한 작업이다.
      y = tf.reshape(y, (1, 1))

      # 최대 64길이 까지 출력으로 받을 것이다.
      seq = tf.TensorArray(tf.int32, 64)

      # tf.keras.Model에 의해서 call 함수는 auto graph모델로 변환이 되게 되는데,
      # 이때, tf.range를 사용해 for문이나 while문을 작성시 내부적으로 tf 함수로 되어있다면
      # 그 for문과 while문이 굉장히 효율적으로 된다.
      for idx in tf.range(64):
        y, h, c = self.dec([y, h, c, H])
        # 아래 두가지 작업은 test data를 예측하므로 처음 예측한값을 다시 다음 step의 입력으로 넣어주어야하기에 해야하는 작업이다.
        # 위의 출력으로 나온 y는 softmax를 지나서 나온 값이므로
        # 가장 높은 값의 index값을 tf.int32로 형변환해주고
        # 위에서 만들어 놓았던 TensorArray에 idx에 y를 추가해준다.
        y = tf.cast(tf.argmax(y, axis=-1), dtype=tf.int32)
        # 위의 값을 그대로 넣어주게 되면 Dimension이 하나밖에 없어서
        # 실제로 네트워크를 사용할 때 Batch를 고려해서 사용해야 하기 때문에 (1,1)으로 설정해 준다.
        y = tf.reshape(y, (1, 1))
        seq = seq.write(idx, y)

        if y == self.eos:
          break
      # stack은 그동안 TensorArray로 받은 값을 쌓아주는 작업을 한다.    
      return tf.reshape(seq.stack(), (1, 64))

# Implement training loop
@tf.function
def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_accuracy):
  # output_labels는 실제 output과 비교하기 위함
  # shifted_labels는 Decoder부분에 입력을 넣기 위함
  output_labels = labels[:, 1:]
  shifted_labels = labels[:, :-1]
  with tf.GradientTape() as tape:
    predictions = model([inputs, shifted_labels], training=True)
    loss = loss_object(output_labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)

  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)
  train_accuracy(output_labels, predictions)

# Implement algorithm test
@tf.function
def test_step(model, inputs):
  return model(inputs, training=False)