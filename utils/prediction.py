from transformers import pipeline

# Load fake news detection model
classifier = pipeline(
    "text-classification",
    model="hamzab/roberta-fake-news-classification",
    tokenizer="hamzab/roberta-fake-news-classification"
)

def predict_news(text, model_type="bert"):

    result = classifier(text)[0]

    label = result["label"].lower()
    score = result["score"]

    if "fake" in label:
        prediction = "FAKE NEWS"
        risk = "High Risk"
    else:
        prediction = "REAL NEWS"
        risk = "Low Risk"

    confidence = round(score * 100, 2)

    reliability = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"

    sentiment = "Neutral"

    keywords = text.split()[:5]

    short_flag = len(text.split()) < 20

    return prediction, confidence, reliability, risk, sentiment, keywords, short_flag