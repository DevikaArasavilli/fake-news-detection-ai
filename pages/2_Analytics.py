import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="Analytics",
    page_icon="📊",
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

/* Navigation spacing */
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
font-size:120px;
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

/* Analytics cards */
.card{
padding:25px;
border-radius:12px;
background:#1f2937;
color:white;
text-align:center;
box-shadow:0px 4px 10px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# =============================
# Header
# =============================
st.markdown(
'<h1 class="main-title">📊 Dataset Analytics</h1>',
unsafe_allow_html=True
)

st.markdown(
'<p class="subtitle">Explore the dataset used to train the Fake News Detection models.</p>',
unsafe_allow_html=True
)

st.divider()

# =============================
# Dataset Stats (Simulated)
# =============================

fake_rows = 23481
real_rows = 21417
total_rows = fake_rows + real_rows

col1, col2, col3 = st.columns(3)

col1.markdown(f"""
<div class="card">
<h2>{total_rows}</h2>
<p>Total Articles</p>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="card">
<h2>{fake_rows}</h2>
<p>Fake News</p>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="card">
<h2>{real_rows}</h2>
<p>Real News</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# =============================
# Distribution Chart
# =============================

st.write("### Fake vs Real Distribution")

labels = ["Fake", "Real"]
values = [fake_rows, real_rows]

fig, ax = plt.subplots(figsize=(4,3))

ax.bar(labels, values)

ax.set_ylabel("Number of Articles")

st.pyplot(fig)

st.divider()

# =============================
# Example Articles
# =============================

st.write("### Example News Articles")

sample = pd.DataFrame({
    "Article": [
        "Government introduces new economic reform bill",
        "Scientists discover promising cancer treatment",
        "Celebrity falsely claims election fraud",
        "Viral social media post spreads misinformation",
        "Stock markets reach record highs"
    ],
    "Label": ["Real", "Real", "Fake", "Fake", "Real"]
})

st.dataframe(sample)