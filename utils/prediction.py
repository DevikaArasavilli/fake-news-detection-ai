from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load model from HuggingFace
MODEL_NAME = "distilbert-base-uncased"

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()


def predict_news(text, model_type="bert"):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    fake_prob = probs[0][0].item()

    if fake_prob > 0.5:
        prediction = "FAKE NEWS"
    else:
        prediction = "REAL NEWS"

    confidence = round(abs(fake_prob - 0.5) * 200, 2)

    reliability = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"

    risk = "High Risk" if prediction == "FAKE NEWS" else "Low Risk"

    sentiment = "Neutral"

    keywords = text.split()[:5]

    short_flag = len(text.split()) < 20

    return prediction, confidence, reliability, risk, sentiment, keywords, short_flag