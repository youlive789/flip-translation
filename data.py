from embedding import Embedding
from multiprocessing import Pool
from functools import partial
import time

class Dataset:
    def __init__(self, english, korean):
        pass

    def get_corpus(self):
        pass

    def prepare_dataset(self):
        pass

    def _tokenize_dataset(self):
        pass

    def _fix_length_dataset(self):
        pass
    
def fix(sentence):
    length = len(sentence)
    if length < 16:
        for i in range(16 - length):
            sentence.append("<PAD>")
    elif length > 16:
        sentence = sentence[:16]
    return sentence

def emb(sentence, embedding):
    return [embedding.word_to_vec(word) for word in sentence]

if __name__ == "__main__":

    # 데이터 로드
    with open("data/train.en", encoding="utf-8") as f:
        train_english = f.readlines()

    with open("data/train.ko", encoding="utf-8") as f:
        train_korean = f.readlines()
    
    embedding = Embedding(corpus=None, word_train=False)

    # 문서 토크나이즈
    t = time.time()
    with Pool(2) as pool:
        train_english = pool.map(embedding.tokenize, train_english)
        train_korean = pool.map(embedding.tokenize, train_korean)
    print(train_english[0])
    print(train_korean[0])
    print(time.time() - t)

    print()

    # 문장 길이 FIXING - <PAD>
    t = time.time()
    with Pool(2) as pool:
        train_english = pool.map(fix, train_english)
        train_korean = pool.map(fix, train_korean)
    print(train_english[0])
    print(train_korean[0])
    print(time.time() - t)

    print()
    
    t = time.time()
    with Pool(2) as pool:
        train_english = pool.map(partial(emb, embedding=embedding), train_english)
        train_korean = pool.map(partial(emb, embedding=embedding), train_korean)
    print(train_english[0])
    print(train_korean[0])
    print(time.time() - t)