# preprocess.py
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

def preprocess_text(texts):
    # Contoh fungsi untuk preprocessing teks
    stop_words = set(stopwords.words('english'))
    processed_texts = []

    for text in texts:
        # Tokenisasi dan pembersihan teks
        tokens = word_tokenize(text.lower())  # Tokenisasi dan ubah menjadi lowercase
        tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
        processed_texts.append(' '.join(tokens))

    return processed_texts
