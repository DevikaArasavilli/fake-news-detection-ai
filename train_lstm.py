import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utils.preprocessing import clean_text

# ==========================
# Configuration
# ==========================
MAX_WORDS = 20000
MAX_LEN = 300
RANDOM_STATE = 42

MODEL_DIR = "model/lstm"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================
# Load Dataset
# ==========================
print("Loading dataset...")

fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], axis=0)

# Combine title + text (better performance)
df["content"] = df["title"] + " " + df["text"]

# Shuffle dataset
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# ==========================
# Preprocess Text
# ==========================
print("Cleaning text...")

df["content"] = df["content"].apply(clean_text)

texts = df["content"]
labels = df["label"]

# ==========================
# Train/Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=RANDOM_STATE
)

# ==========================
# Tokenization
# ==========================
print("Tokenizing text...")

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post")

# ==========================
# Build LSTM Model
# ==========================
print("Building LSTM model...")

model = Sequential([
    Embedding(MAX_WORDS, 128),
    LSTM(128),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# ==========================
# Callbacks
# ==========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "fake_news_model.keras"),
    monitor="val_loss",
    save_best_only=True
)

# ==========================
# Train Model
# ==========================
print("Training model...")

history = model.fit(
    X_train_pad,
    y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=64,
    callbacks=[early_stop, checkpoint]
)

# ==========================
# Evaluate Model
# ==========================
print("Evaluating model...")

pred_probs = model.predict(X_test_pad)
preds = (pred_probs > 0.5).astype(int)

print("\nAccuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n")
print(classification_report(y_test, preds))

# ==========================
# Save Tokenizer
# ==========================
with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

print("\nModel and tokenizer saved successfully.")