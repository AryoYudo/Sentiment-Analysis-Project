import pandas as pd
import numpy as np  # Impor numpy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
df = pd.read_csv("data/imdb_reviews.csv")  # Sesuaikan dengan path file CSV

# Tokenisasi dan padding teks
tokenizer = Tokenizer(num_words=5000)  # Membatasi jumlah kata unik
tokenizer.fit_on_texts(df['review'].values)
X = tokenizer.texts_to_sequences(df['review'].values)
X = pad_sequences(X, maxlen=100)  # Sesuaikan dengan panjang urutan yang diinginkan

# Mengonversi label menjadi format yang sesuai
y = df['sentiment'].values  # Misalnya, jika sentiment berupa teks seperti 'positive' dan 'negative'
y = [1 if sentiment == 'positive' else 0 for sentiment in y]  # Mengonversi menjadi 1 untuk positif dan 0 untuk negatif

# Ubah X dan y menjadi array NumPy
X = np.array(X)
y = np.array(y)

# Definisikan model LSTM
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=100, input_length=100))  # Sesuaikan dengan dimensi data
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

# Kompilasi model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callback untuk menghentikan pelatihan jika model sudah tidak meningkat
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Latih model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Simpan model
model.save('sentiment_model.keras')
