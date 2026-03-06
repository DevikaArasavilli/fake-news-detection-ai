import streamlit as st
from transformers import pipeline
import os


# ==============================
# Load Local DistilBERT Model
# ==============================
@st.cache_resource
def load_model():

    model_path = os.path.join("model", "distilbert")

    classifier = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path
    )

    return classifier


# ==============================
# Fake News Prediction
# ==============================
def predict_news(text, model_type="distilbert"):

    classifier = load_model()

    # Limit input length
    text = text[:800]

    result = classifier(text)[0]

    label = result["label"]
    score = result["score"]

    # ------------------------------
    # Label Mapping
    # ------------------------------
    # Most fake-news models use:
    # LABEL_0 = FAKE
    # LABEL_1 = REAL

    if label == "LABEL_0":
        prediction = "FAKE NEWS"
        risk = "High Risk"
    else:
        prediction = "REAL NEWS"
        risk = "Low Risk"

    # ------------------------------
    # Confidence Score
    # ------------------------------
    confidence = round(score * 100, 2)

    if confidence > 80:
        reliability = "High"
    elif confidence > 60:
        reliability = "Medium"
    else:
        reliability = "Low"

    # ------------------------------
    # Sentiment (simple)
    # ------------------------------
    sentiment = "Neutral"

    positive_words = ["growth", "success", "development", "benefit"]
    negative_words = ["crisis", "attack", "fraud", "corruption"]

    lower_text = text.lower()

    if any(word in lower_text for word in positive_words):
        sentiment = "Positive"

    if any(word in lower_text for word in negative_words):
        sentiment = "Negative"

    # ------------------------------
    # Keyword Extraction
    # ------------------------------
    words = text.split()
    keywords = words[:5]

    # ------------------------------
    # Short Text Flag
    # ------------------------------
    short_flag = len(words) < 20

    return prediction, confidence, reliability, risk, sentiment, keywords, short_flag