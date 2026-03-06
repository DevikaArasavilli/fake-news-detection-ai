import os
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from transformers import pipeline

from utils.preprocessing import clean_text
from utils.explainability import get_sentiment, extract_keywords


# -------------------------------
# Load LSTM model + tokenizer
# -------------------------------
@st.cache_resource
def load_lstm():
    model_path = os.path.join("model", "lstm", "fake_news_model.keras")
    tokenizer_path = os.path.join("model", "lstm", "tokenizer.pkl")

    model = load_model(model_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


# -------------------------------
# Load HuggingFace model
# -------------------------------
@st.cache_resource
def load_transformer():
    clf = pipeline(
        "text-classification",
        model="hamzab/roberta-fake-news-classification",
        tokenizer="hamzab/roberta-fake-news-classification"
    )
    return clf


# -------------------------------
# LSTM prediction
# -------------------------------
def lstm_predict(text):

    model, tokenizer = load_lstm()

    text = clean_text(text)

    seq = tokenizer.texts_to_sequences([text])

    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = pad_sequences(seq, maxlen=300)

    prob = model.predict(seq)[0][0]

    if prob > 0.5:
        return "REAL", prob
    else:
        return "FAKE", 1 - prob


# -------------------------------
# Transformer prediction
# -------------------------------
def transformer_predict(text):

    clf = load_transformer()

    result = clf(text[:800])[0]

    label = result["label"].lower()
    score = result["score"]

    if "fake" in label:
        return "FAKE", score
    else:
        return "REAL", score


# -------------------------------
# Final prediction function
# -------------------------------
def predict_news(text, model_type="ensemble"):

    text = text.strip()

    if model_type.lower() == "lstm":

        label, score = lstm_predict(text)

    elif model_type.lower() == "distilbert":

        label, score = transformer_predict(text)

    else:
        # Ensemble (average decision)

        l_label, l_score = lstm_predict(text)
        t_label, t_score = transformer_predict(text)

        if l_label == t_label:
            label = l_label
            score = (l_score + t_score) / 2
        else:
            # trust LSTM more
            label = l_label
            score = l_score


    # -----------------------
    # Format outputs
    # -----------------------

    if label == "FAKE":
        prediction = "FAKE NEWS"
        risk = "High Risk"
    else:
        prediction = "REAL NEWS"
        risk = "Low Risk"

    confidence = round(score * 100, 2)

    if confidence > 80:
        reliability = "High"
    elif confidence > 60:
        reliability = "Medium"
    else:
        reliability = "Low"

    sentiment = get_sentiment(text)
    keywords = extract_keywords(text)

    short_flag = len(text.split()) < 20

    return prediction, confidence, reliability, risk, sentiment, keywords, short_flag