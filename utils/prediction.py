import os
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.preprocessing import clean_text
from utils.explainability import get_sentiment, extract_keywords


# ==============================
# Load LSTM model (cached)
# ==============================
@st.cache_resource
def load_lstm():

    model_path = os.path.join("model", "lstm", "fake_news_model.keras")
    tokenizer_path = os.path.join("model", "lstm", "tokenizer.pkl")

    model = load_model(model_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


# ==============================
# Prediction function
# ==============================
def predict_news(text, model_type="lstm"):

    model, tokenizer = load_lstm()

    # Clean text
    cleaned = clean_text(text)

    # Convert text → sequence
    seq = tokenizer.texts_to_sequences([cleaned])
    seq = pad_sequences(seq, maxlen=300)

    # Model prediction
    prob = float(model.predict(seq)[0][0])

    # ==============================
    # Improved classification threshold
    # ==============================
    threshold = 0.65

    if prob >= threshold:
        prediction = "REAL NEWS"
        score = prob
        risk = "Low Risk"
    else:
        prediction = "FAKE NEWS"
        score = 1 - prob
        risk = "High Risk"

    # ==============================
    # Confidence (clean value)
    # ==============================
    confidence = round(float(score) * 100, 2)

    # ==============================
    # Reliability
    # ==============================
    if confidence >= 80:
        reliability = "High"
    elif confidence >= 60:
        reliability = "Medium"
    else:
        reliability = "Low"

    # ==============================
    # Sentiment & Keywords
    # ==============================
    sentiment = get_sentiment(text)
    keywords = extract_keywords(text)

    # Short text flag
    short_flag = len(text.split()) < 20

    return prediction, confidence, reliability, risk, sentiment, keywords, short_flag