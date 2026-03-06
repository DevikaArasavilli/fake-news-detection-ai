🧠 AI Fake News Detection Platform

An AI-powered system that detects whether a news article is Fake or Real using advanced Natural Language Processing models.

This project combines Deep Learning (LSTM) and Transformer models (DistilBERT) with an interactive Streamlit dashboard.

🚀 Live Demo

👉 Try the Web App

https://fake-news-detection-ai-ywaih6pl7inlzt3b3wigpa.streamlit.app
🚀 Features

Fake news detection using AI

DistilBERT transformer model

LSTM deep learning model

Ensemble prediction system

Confidence scoring

Risk analysis

Sentiment detection

Keyword extraction

Interactive Streamlit web app

🧠 AI Models Used
1️⃣ LSTM (Long Short-Term Memory)

A deep learning architecture designed to process sequential text data.

2️⃣ DistilBERT

A transformer-based NLP model derived from BERT that understands contextual relationships in text.

3️⃣ Ensemble Model

Combines predictions from LSTM and DistilBERT to improve accuracy.

📊 Dataset

Dataset used:

Fake.csv

True.csv

Total samples: 44,898 news articles

⚙️ AI Pipeline
User Input
   ↓
Text Preprocessing
   ↓
Model Selection (LSTM / DistilBERT / Ensemble)
   ↓
Prediction
   ↓
Confidence & Risk Analysis
🖥️ Application Pages
🏠 Home

Overview of the AI system.

📰 Detector

Predict whether a news article is Fake or Real.

📊 Analytics

Dataset insights and statistics.

🧠 Model Info

Explanation of the AI models and architecture.

🛠️ Technologies Used

Python

Streamlit

TensorFlow / Keras

Transformers (HuggingFace)

PyTorch

Pandas

NumPy

Scikit-learn

NLTK

📂 Project Structure
fake_news_project/

app.py
requirements.txt

data/
Fake.csv
True.csv

model/
distilbert/
lstm/

pages/
1_Detector.py
2_Analytics.py
3_Model_Info.py

utils/
prediction.py
preprocessing.py
explainability.py
▶️ Run Locally

Install dependencies

pip install -r requirements.txt

Run the app

streamlit run app.py
🌐 Deployment

The app is deployed using Streamlit Community Cloud.

👩‍💻 Author

Devika Arasavilli
B.Tech – Artificial Intelligence & Data Science

⭐ If you like this project, give it a star on GitHub.

