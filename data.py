import re
# import tensorflow as tf

"""[reference]
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/nmt_with_attention.ipynb?hl=ko#scrollTo=rd0jw-eC3jEh
"""

class Dataset:
    def __init__(self, dir_source, dir_target):
        self.dir_source = dir_source
        self.dir_target = dir_target

    def _read_files():
        pass

    def _preprocess_sentence(self, sentence):
        # 구두점처리
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        sentence = sentence.strip()
        sentence = "<start> " + sentence + " <end>"
        return sentence

if __name__ == "__main__":
    with open("data/train.ko", encoding="utf-8") as f:
        ko = f.readlines()

    with open("data/train.en", encoding="utf-8") as f:
        en = f.readlines()

    dataset = Dataset("s", "t")
    print(ko[0])
    print(dataset._preprocess_sentence(ko[0]))