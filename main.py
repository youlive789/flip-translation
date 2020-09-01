import numpy as np
import os, sys, getopt
import tensorflow as tf

from data import Dataset
from model import Seq2Seq, train_step, test_step
from embedding import Embedding


if __name__ == "__main__":

    main_dir = sys.argv[1]
    training = sys.argv[2]

    is_encoder_data_exist = os.path.isfile(main_dir + "data/encoder_input.npy")
    is_decoder_data_exist = os.path.isfile(main_dir + "data/decoder_input.npy")
    is_decoer_output_exist = os.path.isfile(main_dir + "data/decoder_output.npy")

    if is_encoder_data_exist and is_decoder_data_exist and is_decoer_output_exist:
        print("임베딩 데이터 존재, 로딩")
        encoder_input = np.load(main_dir + "data/encoder_input.npy")
        decoder_input = np.load(main_dir + "data/decoder_input.npy")
        decoder_output = np.load(main_dir + "data/decoder_output.npy")
        
        embedding = Embedding(corpus=None, word_train=False, main_dir=main_dir)
    else:
        print("데이터 준비 시작")
        with open(main_dir + "data/train.en", encoding="utf-8") as f:
            train_english = f.readlines()

        with open(main_dir + "data/train.ko", encoding="utf-8") as f:
            train_korean = f.readlines()

        if training == "train":
            print("단어학습 시작")
            embedding = Embedding(corpus=train_english + train_korean, word_train=True, main_dir=main_dir)
        else :
            print("단어모델 로드")
            embedding = Embedding(corpus=None, word_train=False, main_dir=main_dir)
        
        dataset = Dataset(train_english, train_korean)
        encoder_input, decoder_input, decoder_output = dataset.prepare_dataset(embedding)
        np.save(main_dir + "data/encoder_input", encoder_input)
        np.save(main_dir + "data/decoder_input", decoder_input)
        np.save(main_dir + "data/decoder_output", decoder_output)

    model = Seq2Seq(16, 100, len(embedding.idx_word_dict.keys()))

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    if training == "train":
        batch_size = 32
        count = int(len(encoder_input) / batch_size)
        for epoch in range(10):
            for c in range(count):
                start = c * batch_size
                end = (c + 1) * batch_size
                train_step(model, encoder_input[start:end], decoder_input[start:end], decoder_output[start:end], loss_object, optimizer, train_loss, train_accuracy)

                template='Epoch {}, Loss: {}, Accuracy:{}'
                print(template.format(epoch + 1,
                                        train_loss.result(),
                                        train_accuracy.result() * 100))

        model.save_weights(main_dir + "saved_model/model")
    else:
        model.load_weights(main_dir + "saved_model/model")

    test_idx = test_step(model, encoder_input[0:1])
    test_idx = test_idx.numpy().tolist()[0]
    result = [embedding.idx_to_word(idx) for idx in test_idx]
    print(result)