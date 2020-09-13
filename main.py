import tensorflow as tf
from data import Dataset
from model import Encoder, Decoder, Seq2seq, train_step, test_step

if __name__ == "__main__":
    dataset = Dataset("data/train.en", "data/train.ko")
    en_tensor, en_tokenizer, ko_tensor, ko_tokenizer = dataset.load_dataset()
    en_words_count = len(en_tokenizer.word_index)
    ko_words_count = len(ko_tokenizer.word_index)
    print(en_words_count, ko_words_count)

    train_ds = tf.data.Dataset.from_tensor_slices((en_tensor, ko_tensor)).shuffle(10000).batch(2)
    model = Seq2seq(source_words_count=en_words_count, target_words_count=ko_words_count,
        sos=ko_tokenizer.word_index["<start>"], eos=ko_tokenizer.word_index["<end>"])

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    for epoch in range(1):
        for seqs, labels in train_ds:
            train_step(model, seqs, labels, loss_object, optimizer, train_loss, train_accuracy)

            template='Epoch {}, Loss: {}, Accuracy:{}'
            print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100))