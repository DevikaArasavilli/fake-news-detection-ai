import torch
import torch.nn.functional as F
import numpy as np
import pickle

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.preprocessing import clean_text, extract_keywords


# =============================
# Constants
# =============================
MAX_LEN = 300


# =============================
# Load LSTM Model
# =============================
lstm_model = load_model("model/lstm/fake_news_model.keras")

with open("model/lstm/tokenizer.pkl", "rb") as f:
    lstm_tokenizer = pickle.load(f)


# =============================
# Load DistilBERT Model
# =============================
bert_model = DistilBertForSequenceClassification.from_pretrained("model/distilbert")
bert_tokenizer = DistilBertTokenizer.from_pretrained("model/distilbert")

bert_model.eval()


# =============================
# Simple Sentiment Detection
# =============================
def simple_sentiment(text):

    positive_words = ["good", "growth", "success", "positive", "benefit"]
    negative_words = ["fake", "fraud", "scam", "corrupt", "crime"]

    text = text.lower()

    pos = sum(word in text for word in positive_words)
    neg = sum(word in text for word in negative_words)

    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    else:
        return "Neutral"


# =============================
# LSTM Prediction
# =============================
def predict_lstm(text):

    cleaned = clean_text(text)

    seq = lstm_tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    prob = lstm_model.predict(padded)[0][0]

    prediction = "REAL NEWS" if prob > 0.5 else "FAKE NEWS"

    confidence = prob if prob > 0.5 else 1 - prob

    return prediction, confidence


# =============================
# DistilBERT Prediction
# =============================
def predict_bert(text):

    inputs = bert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)

    confidence = torch.max(probs).item() * 0.97  # reduce overconfidence

    prediction = torch.argmax(probs).item()

    label = "REAL NEWS" if prediction == 1 else "FAKE NEWS"

    return label, confidence


# =============================
# Ensemble Prediction
# =============================
def predict_ensemble(text):

    lstm_pred, lstm_conf = predict_lstm(text)
    bert_pred, bert_conf = predict_bert(text)

    lstm_score = lstm_conf if lstm_pred == "REAL NEWS" else (1 - lstm_conf)
    bert_score = bert_conf if bert_pred == "REAL NEWS" else (1 - bert_conf)

    final_score = (lstm_score + bert_score) / 2

    if final_score > 0.5:
        prediction = "REAL NEWS"
    else:
        prediction = "FAKE NEWS"

    confidence = final_score if prediction == "REAL NEWS" else (1 - final_score)

    return prediction, confidence


# =============================
# Main Prediction Pipeline
# =============================
def predict_news(text, model_type="bert"):

    short_flag = len(text.split()) < 20

    if model_type == "lstm":
        prediction, confidence = predict_lstm(text)

    elif model_type == "bert":
        prediction, confidence = predict_bert(text)

    else:
        prediction, confidence = predict_ensemble(text)

    confidence_percent = round(confidence * 100, 2)

    if confidence_percent > 85:
        reliability = "High"
    elif confidence_percent > 65:
        reliability = "Medium"
    else:
        reliability = "Low"

    risk = "High Risk" if prediction == "FAKE NEWS" else "Low Risk"

    sentiment = simple_sentiment(text)

    keywords = extract_keywords(text)

    return (
        prediction,
        confidence_percent,
        reliability,
        risk,
        sentiment,
        keywords,
        short_flag
    )