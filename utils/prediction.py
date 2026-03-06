from transformers import pipeline
import streamlit as st
from utils.explainability import get_sentiment, extract_keywords


# ==============================
# Load Model (cached)
# ==============================
@st.cache_resource
def load_model():

    classifier = pipeline(
        "text-classification",
        model="hamzab/roberta-fake-news-classification",
        tokenizer="hamzab/roberta-fake-news-classification"
    )

    return classifier


# ==============================
# Prediction Function
# ==============================
def predict_news(text, model_type="distilbert"):

    classifier = load_model()

    # limit very long text
    text = text[:800]

    result = classifier(text)[0]

    label = result["label"]
    score = result["score"]

    # ==============================
    # Label Mapping
    # ==============================
    if label in ["FAKE", "LABEL_0"]:
        prediction = "FAKE NEWS"
        risk = "High Risk"
    else:
        prediction = "REAL NEWS"
        risk = "Low Risk"

    # ==============================
    # Confidence (FIXED ROUNDING)
    # ==============================
    confidence = round(score * 100, 2)

    # ==============================
    # Reliability Level
    # ==============================
    if confidence > 80:
        reliability = "High"
    elif confidence > 60:
        reliability = "Medium"
    else:
        reliability = "Low"

    # ==============================
    # Sentiment + Keywords
    # ==============================
    sentiment = get_sentiment(text)
    keywords = extract_keywords(text)

    # short text warning
    short_flag = len(text.split()) < 20

    return prediction, confidence, reliability, risk, sentiment, keywords, short_flag