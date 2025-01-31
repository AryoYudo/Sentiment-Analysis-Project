from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

MAX_NB_WORDS = 10000  # Batas jumlah kata
MAX_SEQUENCE_LENGTH = 100  # Panjang maksimum setiap urutan

def preprocess_text(reviews):
    # Tokenisasi dan pembersihan teks
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)

    # Pastikan tidak ada nilai None dalam sequences
    sequences = [seq if seq is not None else [] for seq in sequences]

    # Padding sequences untuk memastikan panjangnya konsisten
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    return padded_sequences
