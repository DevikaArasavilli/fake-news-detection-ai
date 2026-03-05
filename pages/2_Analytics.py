import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Analytics",
    page_icon="📊",
    layout="wide"
)
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

/* Remove empty space */
[data-testid="stSidebar"] > div:first-child {
    padding-bottom:10px;
}

</style>
""", unsafe_allow_html=True)

# =============================
# Page Heading
# =============================
st.title("📊 Dataset Analytics")

st.write("Explore the dataset used to train the Fake News Detection models.")

st.divider()

# =============================
# Load Dataset
# =============================
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = "Fake"
true["label"] = "Real"

df = pd.concat([fake, true])

# =============================
# Dataset Metrics
# =============================
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Articles", len(df))
col2.metric("Fake News", len(fake))
col3.metric("Real News", len(true))

st.divider()

# =============================
# Dataset Distribution Chart
# =============================
st.subheader("Fake vs Real News Distribution")

counts = df["label"].value_counts()

fig, ax = plt.subplots(figsize=(4,3))   # smaller chart size

ax.bar(counts.index, counts.values)

ax.set_xlabel("News Type")
ax.set_ylabel("Number of Articles")

st.pyplot(fig)

st.caption("This chart shows how many fake and real news articles exist in the dataset.")

st.divider()

# =============================
# Article Length Insight
# =============================
st.subheader("Average Article Length")

df["length"] = df["text"].apply(len)

avg_fake = fake["text"].apply(len).mean()
avg_real = true["text"].apply(len).mean()

col1, col2 = st.columns(2)

col1.metric("Average Fake News Length", int(avg_fake))
col2.metric("Average Real News Length", int(avg_real))

st.divider()

# =============================
# Example Articles
# =============================
st.subheader("Sample Articles from Dataset")

st.dataframe(
    df.sample(5)[["title","label"]],
    height=180
)

st.caption("Randomly sampled articles from the dataset used for model training.")