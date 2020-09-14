import io
import re
import tensorflow as tf

"""[reference]
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/nmt_with_attention.ipynb?hl=ko#scrollTo=rd0jw-eC3jEh
https://heung-bae-lee.github.io/2020/01/22/deep_learning_11/
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

    def load_dataset(self, num_words):
        source_data, target_data = self.create_dataset()
        input_tensor, input_tokenizer = self._tokenize(source_data, num_words)
        target_tensor, target_tokenizer = self._tokenize(target_data, num_words)
        return input_tensor, input_tokenizer, target_tensor, target_tokenizer

    def _preprocess_sentence(self, sentence):
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        sentence = sentence.strip()
        sentence = "<start> " + sentence + " <end>"
        return sentence

    def _tokenize(self, lang, num_words):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, filters='')
        lang_tokenizer.fit_on_texts(lang)

        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

        return tensor, lang_tokenizer

if __name__ == "__main__":

    dataset = Dataset("data/train.en", "data/train.ko")
    en, ko = dataset.create_dataset()
    en_tensor, en_tokenizer, ko_tensor, ko_tokenizer = dataset.load_dataset()
    print(en_tensor[-1], len(en_tensor[-1]))
    print(en_tokenizer.sequences_to_texts(en_tensor[-1:]))
    print()