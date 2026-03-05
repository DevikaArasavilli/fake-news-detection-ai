from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def extract_keywords(text, top_n=5):
    words = text.split()
    common_words = Counter(words).most_common(top_n)
    return [word for word, _ in common_words]