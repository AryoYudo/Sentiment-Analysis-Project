import tensorflow as tf
from preprocess import clean_text, preprocess_text, tokenizer

# 1️⃣ Load Model yang Sudah Dilatih
model = tf.keras.models.load_model("../models/model_sentiment_lstm.h5")

def predict_sentiment(review):
    """ Memprediksi sentimen dari sebuah review """
    cleaned_review = clean_text(review)
    processed_review = preprocess_text([cleaned_review])
    prediction = model.predict(processed_review)
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    return sentiment

# 2️⃣ Contoh Penggunaan
if __name__ == "__main__":
    review_text = input("Masukkan review: ")
    print("Prediksi Sentimen:", predict_sentiment(review_text))
