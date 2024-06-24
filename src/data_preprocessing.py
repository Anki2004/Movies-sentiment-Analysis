import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data['review'], data['sentiment']


def data_preprocess(X, y, max_words = 10000, max_len = 100):

    y = (y == 'positive').astype(int)
    tokenizer = Tokenizer(num_words = max_words)
    tokenizer.fit_on_texts(X)

    X_seq = tokenizer.texts_to_sequences(X)

    X_pad = pad_sequences(X_seq, maxlen = max_len)
    return X_pad, y, tokenizer

def prepare_data(file_path, max_words = 10000, max_len = 1000):
    X, y = load_data(file_path)
    X_pad, y, tokenizer = data_preprocess(X, y, max_words, max_len)
    X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, tokenizer

