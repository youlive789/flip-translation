import io
import re
import tensorflow as tf

"""[reference]
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/nmt_with_attention.ipynb?hl=ko#scrollTo=rd0jw-eC3jEh
"""

class Dataset:
    def __init__(self, dir_source, dir_target):
        self.dir_source = dir_source
        self.dir_target = dir_target

    def create_dataset(self):
        source = io.open(self.dir_source, encoding='UTF-8').read().strip().split('\n')
        target = io.open(self.dir_target, encoding='UTF-8').read().strip().split('\n')

        source = [self._preprocess_sentence(sentence) for sentence in source]
        target = [self._preprocess_sentence(sentence) for sentence in target]

        return source, target

    def load_dataset(self):
        en, ko = self.create_dataset()
        input_tensor, input_tokenizer = self._tokenize(en)
        target_tensor, target_tokenizer = self._tokenize(ko)
        return input_tensor, input_tokenizer, target_tensor, target_tokenizer

    def _preprocess_sentence(self, sentence):
        # 구두점처리
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        sentence = sentence.strip()
        sentence = "<start> " + sentence + " <end>"
        return sentence

    def _tokenize(self, lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        lang_tokenizer.fit_on_texts(lang)

        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

        return tensor, lang_tokenizer

if __name__ == "__main__":

    dataset = Dataset("data/train.en", "data/train.ko")
    en, ko = dataset.create_dataset()
    print(ko[0])

    en_tensor, en_tokenizer, ko_tensor, ko_tokenizer = dataset.load_dataset()
    print(en_tensor[0])

    BUFFER_SIZE = len(en_tensor)
    BATCH_SIZE = 64
    steps_per_epoch = len(en_tensor) // BATCH_SIZE
    embedding_dim = 256
    units = 1024
    vocab_en_size = len(en_tokenizer.word_index) + 1
    vocab_ko_size = len(ko_tokenizer.word_index) + 1

    train_dataset = tf.data.Dataset.from_tensor_slices((en_tensor, ko_tensor)).shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

    example_input_batch, example_target_batch = next(iter(train_dataset))
    print(example_input_batch.shape, example_target_batch.shape)
