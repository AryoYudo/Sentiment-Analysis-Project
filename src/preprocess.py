from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import re

# Fungsi untuk membersihkan teks
def clean_text(text):
    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Hapus angka
    text = re.sub(r'\d+', '', text)

    # Hapus karakter non-alphabet (termasuk karakter selain huruf dan spasi)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text

def preprocess_text(texts):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()  # Inisialisasi stemmer
    processed_texts = []

    for text in texts:
        # Bersihkan teks terlebih dahulu
        clean = clean_text(text)

        # Tokenisasi dan ubah ke lowercase
        tokens = word_tokenize(clean.lower())

        # Hapus stopwords, tanda baca, dan kata yang terlalu pendek (misalnya kata dengan 1 karakter)
        tokens = [
            word for word in tokens 
            if word not in stop_words 
            and word not in string.punctuation 
            and len(word) > 1  # Menghindari kata dengan satu karakter
        ]

        # Stemming setiap kata
        tokens = [stemmer.stem(word) for word in tokens]

        # Gabungkan kata-kata yang telah diproses menjadi satu string
        processed_texts.append(' '.join(tokens))

    return processed_texts
