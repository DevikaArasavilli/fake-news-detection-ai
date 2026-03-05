import streamlit as st

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Model Info",
    page_icon="🧠",
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

/* Sidebar navigation */
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
font-size:160px;
font-weight:900;
text-align:center;
color:white;
margin-bottom:10px;
}

/* Subtitle */
.subtitle{
font-size:30px;
text-align:center;
color:#555;
margin-bottom:40px;
}

/* Model cards */
.card{
background:#1f2937;
padding:25px;
border-radius:12px;
color:white;
box-shadow:0 6px 18px rgba(0,0,0,0.25);
transition:0.3s;
height:230px;
}

/* Hover animation */
.card:hover{
transform:translateY(-8px) scale(1.02);
box-shadow:0 12px 25px rgba(0,0,0,0.35);
}

/* Pipeline step */
.pipeline-box{
background:#f4f6fb;
color:#111;        /* Force dark text */
padding:20px;
border-radius:10px;
text-align:center;
font-size:18px;
font-weight:600;
}

</style>
""", unsafe_allow_html=True)

# =============================
# Header
# =============================
st.markdown(
'<h1 class="main-title">🧠 AI Model Architecture</h1>',
unsafe_allow_html=True
)

st.markdown(
'<p class="subtitle">Understanding the AI models powering the Fake News Detection System</p>',
unsafe_allow_html=True
)

st.divider()

# =============================
# Models Section
# =============================
st.write("## Models Used")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
    <h3>1️⃣ LSTM Model</h3>
    <p>
    Long Short-Term Memory (LSTM) is a deep learning architecture
    designed to process sequential text data. It captures patterns
    between words in news articles to classify them as real or fake.
    </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <h3>2️⃣ DistilBERT</h3>
    <p>
    DistilBERT is a transformer-based NLP model derived from BERT.
    It understands contextual relationships between words and
    improves fake news detection accuracy.
    </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
    <h3>3️⃣ Ensemble System</h3>
    <p>
    The ensemble model combines predictions from both
    LSTM and DistilBERT to increase reliability and
    reduce classification errors.
    </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =============================
# AI Pipeline
# =============================
# =============================
# AI Pipeline
# =============================
# =============================
# AI Pipeline
# =============================
st.write("## AI Processing Pipeline")

c1, a1, c2, a2, c3, a3, c4, a4, c5 = st.columns([2,1,2,1,2,1,2,1,2])

with c1:
    st.markdown('<div class="pipeline-box">User Input</div>', unsafe_allow_html=True)

with a1:
    st.markdown("<h2 style='text-align:center;'>➡</h2>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="pipeline-box">Text Preprocessing</div>', unsafe_allow_html=True)

with a2:
    st.markdown("<h2 style='text-align:center;'>➡</h2>", unsafe_allow_html=True)

with c3:
    st.markdown('<div class="pipeline-box">Model Selection</div>', unsafe_allow_html=True)

with a3:
    st.markdown("<h2 style='text-align:center;'>➡</h2>", unsafe_allow_html=True)

with c4:
    st.markdown('<div class="pipeline-box">AI Prediction</div>', unsafe_allow_html=True)

with a4:
    st.markdown("<h2 style='text-align:center;'>➡</h2>", unsafe_allow_html=True)

with c5:
    st.markdown('<div class="pipeline-box">Confidence + Risk Analysis</div>', unsafe_allow_html=True)

# spacing below pipeline
st.markdown("<br><br>", unsafe_allow_html=True)
# =============================
# Final Info
# =============================
st.success(
"""
This platform integrates classical deep learning (LSTM) and modern
transformer-based NLP models (DistilBERT) to build a robust
fake news detection system capable of analyzing large volumes
of news content.
"""
)