from transformers import pipeline
import streamlit as st
from utils.explainability import get_sentiment, extract_keywords


@st.cache_resource
def load_model():
    classifier = pipeline(
        "text-classification",
        model="hamzab/roberta-fake-news-classification",
        tokenizer="hamzab/roberta-fake-news-classification"
    )
    return classifier


def predict_news(text, model_type="distilbert"):

    classifier = load_model()

    text = text[:800]

    result = classifier(text)[0]

    label = result["label"]
    score = result["score"]

    # FIXED label mapping
    if label in ["FAKE", "LABEL_0"]:
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