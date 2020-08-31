import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from gensim.test.utils import get_tmpfile
from gensim.models.fasttext import FastText

class Embedding:

    MODEL_SAVED_DIR = "saved_model/fasttext.model"
    TOKENIZER_SAVED_DIR = "saved_model/tokenizer.pkl"

    def __init__(self, corpus, word_train:bool, main_dir:str = ""):
        self.MODEL_SAVED_DIR = main_dir + self.MODEL_SAVED_DIR
        self.TOKENIZER_SAVED_DIR = main_dir + self.TOKENIZER_SAVED_DIR

        if not word_train:
            self.fasttext = FastText.load(self.MODEL_SAVED_DIR)
            self._load_tokenizer()
        else:
            self._word_extract(corpus)
            self._set_cohension_score()
            self._set_tokenizer()
            self._save_tokenizer()
            self._word_vec_train([self.tokenize(sentence) for sentence in corpus])

        self.idx_word_dict = dict(zip(np.arange(4, len(self.fasttext.wv.vectors) + 4), self.fasttext.wv.index2word))
        self.idx_word_dict[0] = '<PAD>'
        self.idx_word_dict[1] = '<STA>'
        self.idx_word_dict[2] = '<EOS>'
        self.idx_word_dict[3] = '<UNK>'
        self.word_idx_dict = {v: k for k, v in self.idx_word_dict.items()}

    def _word_extract(self, corpus):
        self.extractor = WordExtractor()
        self.extractor.train(corpus)
        self.words = self.extractor.extract()

    def _set_cohension_score(self):
        self.cohesion_score = {word:score.cohesion_forward for word, score in self.words.items()}

    def _set_tokenizer(self):
        self.tokenizer = LTokenizer(scores=self.cohesion_score)

    def _save_tokenizer(self):
        with open(self.TOKENIZER_SAVED_DIR, "wb") as f:
            pickle.dump(self.tokenizer, f, pickle.HIGHEST_PROTOCOL)

    def _load_tokenizer(self):
        with open(self.TOKENIZER_SAVED_DIR, "rb") as f:
            self.tokenizer = pickle.load(f)

    def _word_vec_train(self, corpus):
        self.fasttext = FastText(sentences=corpus, size=100, window=5, min_count=1, iter=10)
        self.fasttext.save(self.MODEL_SAVED_DIR)

    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)

    def word_to_vec(self, word):
        try :
            if word == '<PAD>': return np.eye(100, dtype=np.float32)[0]
            elif word == '<STA>': return np.eye(100, dtype=np.float32)[1]
            elif word == '<EOS>': return np.eye(100, dtype=np.float32)[2]
            elif word == '<UNK>': return np.eye(100, dtype=np.float32)[3]
            return self.fasttext.wv.word_vec(word)
        except :
            return np.eye(100, dtype=np.float32)[3]

    def vec_to_word(self, vector):
        if np.array_equal(vector, np.eye(100, dtype=np.float32)[0]):   return '<PAD>'
        elif np.array_equal(vector, np.eye(100, dtype=np.float32)[1]): return '<STA>'
        elif np.array_equal(vector, np.eye(100, dtype=np.float32)[2]): return '<EOS>'
        elif np.array_equal(vector, np.eye(100, dtype=np.float32)[3]): return '<UNK>'
        return self.fasttext.wv.most_similar(positive=[vector], topn=1)[0][0]

    def word_to_index(self, word):
        try :
            return self.word_idx_dict[word]
        except :
            return 3

    def idx_to_word(self, idx):
        return self.idx_word_dict[idx]

if __name__ == "__main__":

    # 데이터 로드
    with open("data/train.en", encoding="utf-8") as f:
        train_english = f.readlines()

    with open("data/train.ko", encoding="utf-8") as f:
        train_korean = f.readlines()
                
    corpus = np.array(train_english + train_korean)
    print(corpus.shape)

    # 단어 학습
    embedding = Embedding(corpus, False)
    
    # 한국어 테스트
    tokenized = embedding.tokenize(train_korean[0])
    print(tokenized)
    embedded = [embedding.word_to_vec(token) for token in tokenized]
    reset = [embedding.vec_to_word(emb) for emb in embedded]
    print(reset)

    # 영어 테스트
    tokenized = embedding.tokenize(train_english[0])
    print(tokenized)
    embedded = [embedding.word_to_vec(token) for token in tokenized]
    reset = [embedding.vec_to_word(emb) for emb in embedded]
    print(reset)
