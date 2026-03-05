import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dataset Analytics")
st.write("Explore the dataset used to train the Fake News Detection models.")

# ----------------------------
# Create sample dataset
# ----------------------------

data = {
    "label": ["Fake"]*23481 + ["Real"]*21417
}

df = pd.DataFrame(data)

st.write("### Dataset Size")
st.metric("Total Articles", len(df))

# ----------------------------
# Distribution Chart
# ----------------------------

st.write("### Fake vs Real Distribution")

counts = df["label"].value_counts()

fig, ax = plt.subplots()
ax.bar(counts.index, counts.values)
ax.set_ylabel("Number of Articles")

st.pyplot(fig)

# ----------------------------
# Sample Articles
# ----------------------------

st.write("### Example News Headlines")

sample = pd.DataFrame({
    "News": [
        "Government announces new economic reforms",
        "Scientists discover new cancer treatment",
        "Celebrity spreads false election rumor",
        "Fake news spreads on social media",
        "Stock market hits record high"
    ],
    "Label": ["Real", "Real", "Fake", "Fake", "Real"]
})

st.dataframe(sample)