import tensorflow as tf
from data import Dataset
from model import Encoder, Decoder, BahdanauAttention

if __name__ == "__main__":
    data = Dataset("data/train.en", "data/train.ko")
    