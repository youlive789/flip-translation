import numpy as np
from embedding import Embedding
from multiprocessing import Pool
from functools import partial
import time

class Dataset:
    def __init__(self, english, korean):
        self.english = english
        self.korean = korean

    def prepare_dataset(self, embedding):
        self._tokenize_dataset(embedding)
        self._fix_length_dataset()
        self._embedding_dataset(embedding)
        return np.array(self.english), np.array(self.korean)

    def _tokenize_dataset(self, embedding):
        with Pool(2) as pool:
            self.english = pool.map(embedding.tokenize, self.english)
            self.korean = pool.map(embedding.tokenize, self.korean)

    def _fix_length_dataset(self):
        with Pool(2) as pool:
            self.english = pool.map(fix, self.english)
            self.korean = pool.map(fix, self.korean)

    def _embedding_dataset(self, embedding):
        with Pool(2) as pool:
            self.english = pool.map(partial(embed, embedding=embedding), self.english)
            self.korean = pool.map(partial(embed, embedding=embedding), self.korean)

def fix(sentence):
    length = len(sentence)
    if length < 16:
        for i in range(16 - length):
            sentence.append("<PAD>")
    elif length > 16:
        sentence = sentence[:16]
    return sentence

def embed(sentence, embedding):
    return [embedding.word_to_vec(word) for word in sentence]

if __name__ == "__main__":

    # 데이터 로드
    with open("data/train.en", encoding="utf-8") as f:
        train_english = f.readlines()

    with open("data/train.ko", encoding="utf-8") as f:
        train_korean = f.readlines()
    
    embedding = Embedding(corpus=None, word_train=False)
    dataset = Dataset(train_english, train_korean)
    train_english_embed, train_korean_embed = dataset.prepare_dataset(embedding)
    print(train_english_embed.shape)
    print(train_korean_embed.shape)

    