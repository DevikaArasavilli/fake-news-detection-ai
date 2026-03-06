import streamlit as st
from utils.prediction import predict_news

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Detector",
    page_icon="📰",
    layout="wide"
)

# =============================
# Styling
# =============================
st.markdown("""
<style>

/* Sidebar width */
[data-testid="stSidebar"] {
    min-width:360px;
    max-width:360px;
}

/* Sidebar navigation spacing */
[data-testid="stSidebarNav"] ul li {
    font-size:20px;
    padding:10px 10px;
    margin-bottom:6px;
}

/* Remove bottom empty space */
[data-testid="stSidebar"] > div:first-child {
    padding-bottom:10px;
}

/* Main title */
.main-title{
font-size:55px !important;
font-weight:900;
text-align:center;
color:white !important;
margin-bottom:10px;
}

/* Subtitle */
.subtitle{
font-size:30px;
text-align:center;
color:#555;
margin-bottom:30px;
}

</style>
""", unsafe_allow_html=True)

# =============================
# Header
# =============================
st.markdown(
'<h1 class="main-title">📰 Fake News Detector</h1>',
unsafe_allow_html=True
)

st.markdown(
'<p class="subtitle">Analyze news articles using AI models (DistilBERT, LSTM, or Ensemble).</p>',
unsafe_allow_html=True
)

st.divider()

# =============================
# Model Selection
# =============================
model_choice = st.selectbox(
    "Choose AI Model",
    ["DistilBERT", "LSTM", "Ensemble"]
)

if model_choice == "DistilBERT":
    model_type = "bert"
elif model_choice == "LSTM":
    model_type = "lstm"
else:
    model_type = "ensemble"

# =============================
# Text Input
# =============================
news_input = st.text_area(
    "Enter News Article",
    height=250
)

# =============================
# Analyze Button
# =============================
if st.button("Analyze News"):

    if news_input.strip() == "":
        st.warning("Please enter news text.")
    else:

        with st.spinner("AI analyzing article..."):

            prediction, confidence, reliability, risk, sentiment, keywords, short_flag = predict_news(
                news_input,
                model_type
            )

        st.divider()

        # =============================
        # Result Banner
        # =============================
        if prediction == "REAL NEWS":
            st.success("✅ REAL NEWS DETECTED")
        else:
            st.error("🚨 FAKE NEWS DETECTED")

        # =============================
        # Confidence Bar
        # =============================
        st.write("### Confidence Level")

        # FIXED PROGRESS BAR
        progress_value = min(max(float(confidence) / 100, 0), 1)
        st.progress(progress_value)

        # =============================
        # Metrics
        # =============================
        col1, col2, col3 = st.columns(3)

        col1.metric("Confidence", f"{confidence}%")
        col2.metric("Reliability", reliability)
        col3.metric("Risk Level", risk)

        st.divider()

        # =============================
        # Sentiment
        # =============================
        st.write("### Sentiment Analysis")
        st.info(sentiment)

        # =============================
        # Keywords
        # =============================
        st.write("### Key Indicators")

        if keywords:

            max_cols = min(len(keywords), 5)
            cols = st.columns(max_cols)

            for i, word in enumerate(keywords[:max_cols]):
                cols[i].markdown(
                    f"""
                    <div style="
                    background:#374151;
                    color:white;
                    padding:8px 14px;
                    border-radius:8px;
                    text-align:center;
                    font-size:14px;">
                    {word}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # =============================
        # Warnings
        # =============================
        if short_flag:
            st.warning("⚠ Text is very short. AI models perform better with full news articles.")

        if confidence > 98:
            st.warning("⚠ Extremely high confidence — verify with trusted sources.")

        if confidence < 60:
            st.warning("⚠ Low confidence prediction — model uncertainty detected.")