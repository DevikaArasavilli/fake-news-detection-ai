import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# download stopwords safely
try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()


# =============================
# Text Cleaning
# =============================
def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-z\s]", "", text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# =============================
# Keyword Extraction
# =============================
def extract_keywords(text, n=5):

    words = text.lower().split()

    freq = {}

    for word in words:
        if word not in stop_words:
            freq[word] = freq.get(word, 0) + 1

    sorted_words = sorted(freq, key=freq.get, reverse=True)

    return sorted_words[:n]