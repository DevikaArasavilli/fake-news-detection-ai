from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import pickle
import numpy as np
from utils.preprocessing import clean_text

# =============================
# Load LSTM Model
# =============================

LSTM_MODEL_PATH = "model/lstm/fake_news_model.keras"
TOKENIZER_PATH = "model/lstm/tokenizer.pkl"

lstm_model = load_model(LSTM_MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    lstm_tokenizer = pickle.load(f)


# =============================
# Load DistilBERT Model
# =============================

BERT_PATH = "model/distilbert"

bert_tokenizer = DistilBertTokenizer.from_pretrained(BERT_PATH)
bert_model = DistilBertForSequenceClassification.from_pretrained(BERT_PATH)

bert_model.eval()


# =============================
# LSTM Prediction
# =============================

def predict_lstm(text):

    text = clean_text(text)

    seq = lstm_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=300)

    prob = lstm_model.predict(padded)[0][0]

    return prob


# =============================
# DistilBERT Prediction
# =============================

def predict_bert(text):

    inputs = bert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    fake_prob = probs[0][0].item()

    return fake_prob


# =============================
# Final Prediction Function
# =============================

def predict_news(text, model_type="ensemble"):

    lstm_prob = predict_lstm(text)
    bert_prob = predict_bert(text)

    if model_type == "lstm":
        prob = lstm_prob

    elif model_type == "bert":
        prob = bert_prob

    else:
        prob = (lstm_prob + bert_prob) / 2


    # prediction
    if prob > 0.5:
        label = "FAKE NEWS"
    else:
        label = "REAL NEWS"

    confidence = round(abs(prob - 0.5) * 200, 2)

    # reliability
    if confidence > 80:
        reliability = "High"
    elif confidence > 60:
        reliability = "Medium"
    else:
        reliability = "Low"

    # risk
    if label == "FAKE NEWS":
        risk = "High Risk"
    else:
        risk = "Low Risk"

    sentiment = "Neutral"

    keywords = text.split()[:5]

    short_flag = len(text.split()) < 20

    return label, confidence, reliability, risk, sentiment, keywords, short_flag