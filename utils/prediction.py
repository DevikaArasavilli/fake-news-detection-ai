from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-fake-news-detection"
)

def predict_news(text, model_type="bert"):

    result = classifier(text)[0]

    label = result["label"]
    score = result["score"]

    # Correct label mapping
    if label == "LABEL_0":
        prediction = "FAKE NEWS"
    else:
        prediction = "REAL NEWS"

    confidence = round(score * 100, 2)

    reliability = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"

    risk = "High Risk" if prediction == "FAKE NEWS" else "Low Risk"

    sentiment = "Neutral"

    keywords = text.split()[:5]

    short_flag = len(text.split()) < 20

    return prediction, confidence, reliability, risk, sentiment, keywords, short_flag