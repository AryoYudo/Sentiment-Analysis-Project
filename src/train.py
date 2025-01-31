import pandas as pd
from preprocess import preprocess_text
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.optimizers import Adam

# Load dataset
df = pd.read_csv("data/imdb_reviews.csv")  # Sesuaikan dengan path file CSV

# Periksa dan bersihkan data
print(f"Missing reviews: {df['review'].isnull().sum()}")  # Mengecek nilai kosong
df = df.dropna(subset=['review'])  # Menghapus baris yang memiliki nilai kosong

# Preprocessing teks
X = preprocess_text(df['review'].values)

# Label (sentimen)
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values

# Bangun model LSTM
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))  # Disesuaikan dengan data
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Latih model
model.fit(X, y, batch_size=64, epochs=5, validation_split=0.2)

# Simpan model
model.save('sentiment_model.h5')
