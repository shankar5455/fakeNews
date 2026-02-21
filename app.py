"""
Fake News Sentiment Analysis - Streamlit Web Application
=========================================================
Run with:
    streamlit run app.py
"""

import pickle

import streamlit as st
from textblob import TextBlob

from utils import clean_text


def get_sentiment(text):
    """
    Analyse polarity of *text* using TextBlob and return a labelled emoji string.

    Returns:
        "Positive 😊"  – polarity > 0
        "Negative 😡"  – polarity < 0
        "Neutral  😐"  – polarity == 0
    """
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0:
        return "Positive 😊"
    elif polarity < 0:
        return "Negative 😡"
    return "Neutral 😐"


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the saved Logistic Regression model and TF-IDF vectorizer."""
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


# ─── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)

# ─── UI layout ─────────────────────────────────────────────────────────────────

st.title("📰 Fake News Sentiment Analysis")
st.subheader("A Data-Driven Approach to Understanding Public Emotion")
st.markdown(
    "Paste a news article (headline + body) below and click **Predict** to find out "
    "whether the news is real or fake, along with its emotional tone."
)
st.divider()

news_text = st.text_area(
    "Enter news article text here:",
    placeholder="Paste the news headline and/or article body...",
    height=220,
)

predict_clicked = st.button("🔍 Predict", use_container_width=True)

# ─── Prediction ────────────────────────────────────────────────────────────────

if predict_clicked:
    if not news_text.strip():
        st.error("⚠️ Please enter some text before clicking Predict.")
    else:
        try:
            model, vectorizer = load_model()

            cleaned = clean_text(news_text)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            sentiment = get_sentiment(news_text)

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📋 Prediction Result")
                if prediction == 1:
                    st.error("🚨 **FAKE NEWS**\nThis article is likely fabricated or misleading.")
                else:
                    st.success("✅ **REAL NEWS**\nThis article appears to be credible.")

            with col2:
                st.markdown("### 💬 Sentiment Analysis")
                st.info(f"**Detected Sentiment:** {sentiment}")

        except FileNotFoundError:
            st.error(
                "🔧 Model files not found (`model.pkl` / `vectorizer.pkl`). "
                "Please run `python model_training.py` first to train and save the model."
            )
