import yaml
import argparse
import numpy as np
import tensorflow as tf
from data import Dataset
from model import Encoder, Decoder, Seq2seq, train_step, test_step

"""
[config.yaml example]

train:
  epochs: 1
  num_words: 10
  batch_size: 1
  source_data_path: data/train.en
  target_data_path: data/train.ko
  save_model_path: saved_model/
test:
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train", help="training", type=bool)
    parser.add_argument("config", help="config file path", type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if args.train:
            config = config["train"]
        else:
            config = config["test"]

    if args.train:
        dataset = Dataset(config["source_data_path"], config["target_data_path"])
        en, ko = dataset.create_dataset()
        en_tensor, en_tokenizer, ko_tensor, ko_tokenizer = dataset.load_dataset(config["num_words"])
        en_words_count = len(en_tokenizer.word_index) + 1
        ko_words_count = len(ko_tokenizer.word_index) + 1

        train_ds = tf.data.Dataset.from_tensor_slices((en_tensor, ko_tensor)).shuffle(10000).batch(config["batch_size"])
        model = Seq2seq(source_words_count=en_words_count, target_words_count=ko_words_count,
            sos=ko_tokenizer.word_index["<start>"], eos=ko_tokenizer.word_index["<end>"])

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        for epoch in range(config["epochs"]):
            for seqs, labels in train_ds:
                train_step(model, seqs, labels, loss_object, optimizer, train_loss, train_accuracy)
                template='Epoch {}, Loss: {}, Accuracy:{}'
                print(template.format(epoch, train_loss.result(), train_accuracy.result() * 100))

        print()
        print(en[0])
        test = en_tokenizer.texts_to_sequences(en[0:1])
        pred = test_step(model,np.array(test))
        print(pred)
        print(ko_tokenizer.sequences_to_texts(pred.numpy()))
    else:
        pass