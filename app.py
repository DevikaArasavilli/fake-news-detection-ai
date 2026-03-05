import streamlit as st

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Fake News Detection Platform",
    page_icon="🧠",
    layout="wide"
)

# =============================
# Custom CSS
# =============================
st.markdown("""
<style>

/* =============================
   Sidebar width
============================= */

[data-testid="stSidebar"] {
    min-width:360px;
    max-width:360px;
}


/* =============================
   Sidebar navigation spacing
============================= */

[data-testid="stSidebarNav"] ul {
    padding-top:8px;
}

[data-testid="stSidebarNav"] ul li {
    font-size:20px;
    padding:10px 10px;
    margin-bottom:6px;
    border-radius:8px;
}


/* Remove bottom empty space */

[data-testid="stSidebar"] > div:first-child {
    padding-bottom:10px;
}


/* =============================
   Page Main Title
============================= */
.main-title{
font-size:50px !important;
font-weight:900;
text-align:center;
background: linear-gradient(90deg,#ff4b4b,#6a5acd);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
margin-bottom:10px;
}

.subtitle{
font-size:26px !important;
text-align:center;
color:#555;
margin-bottom:40px;
}


/* =============================
   Feature Cards
============================= */

.card{
padding:28px;
border-radius:16px;
background:#e4ebf7;
color:#111;
box-shadow:0 4px 14px rgba(0,0,0,0.1);
margin-bottom:20px;
transition:0.25s;
}

.card:hover{
transform:scale(1.03);
background:#d8e3f6;
}


.feature-title{
font-size:22px;
font-weight:700;
margin-bottom:12px;
color:#000;
}

</style>
""", unsafe_allow_html=True)


# =============================
# Header Section
# =============================
st.markdown(
"""
<h1 class="main-title">🧠 AI Fake News Detection Platform</h1>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class="subtitle">Powered by LSTM + DistilBERT Ensemble AI Models</div>
""",
unsafe_allow_html=True
)

st.divider()


# =============================
# Metrics Section
# =============================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Models Used", "3")

with col2:
    st.metric("Dataset Size", "44,898")

with col3:
    st.metric("AI Type", "NLP")


st.divider()


# =============================
# Features Section
# =============================
st.subheader("🚀 Platform Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
    <div class="feature-title">🤖 Transformer Detection</div>
    Uses DistilBERT transformer model to understand contextual meaning in news articles.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <div class="feature-title">🧠 Deep Learning Model</div>
    LSTM neural network trained on thousands of real and fake news samples.
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
    <div class="feature-title">⚡ Ensemble AI System</div>
    Combines predictions from multiple models to increase detection reliability.
    </div>
    """, unsafe_allow_html=True)


col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div class="card">
    <div class="feature-title">📊 Confidence Analysis</div>
    Displays prediction confidence levels and reliability indicators.
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="card">
    <div class="feature-title">⚠ Risk Detection</div>
    Flags potential misinformation risks based on model prediction.
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div class="card">
    <div class="feature-title">💬 Sentiment Detection</div>
    Analyzes emotional tone within the news content.
    </div>
    """, unsafe_allow_html=True)


st.divider()


# =============================
# Navigation Info
# =============================
st.info(
"""
Use the **sidebar menu** to navigate:

📰 Detector → Fake news prediction  
📊 Analytics → Dataset insights  
🧠 Model Info → AI model explanation
"""
)


# =============================
# Footer
# =============================
st.caption(
"AI Fake News Detection Platform | Built with Streamlit"
)