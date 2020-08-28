import tensorflow as tf

from data import Dataset
from model import Seq2Seq, train_step, test_step
from embedding import Embedding


if __name__ == "__main__":

    with open("data/train.en", encoding="utf-8") as f:
        train_english = f.readlines()

    with open("data/train.ko", encoding="utf-8") as f:
        train_korean = f.readlines()

    embedding = Embedding(corpus=None, word_train=False)
    dataset = Dataset(train_english, train_korean)
    encoder_input, decoder_input, decoder_output = dataset.prepare_dataset(embedding)

    model = Seq2Seq(16, 100, len(embedding.idx_word_dict.keys()))

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizer.Adam()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    for epoch in range(EPOCHS):
        for seqs, labels in train_ds:
            train_step(model, encoder_input, decoder_input, decoder_output, loss_object, optimizer, train_loss, train_accuracy)

        template='Epoch {}, Loss: {}, Accuracy:{}'
        print(template.format(epoch + 1,
                                train_loss.result(),
                                train_accuracy.result() * 100))