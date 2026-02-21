"""
Fake News Sentiment Analysis - Model Training Script
=====================================================
This script:
1. Loads and preprocesses the fake_or_real_news.csv dataset
2. Trains a Logistic Regression model using TF-IDF features
3. Evaluates model performance
4. Saves the trained model and vectorizer using pickle
"""

import pickle
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import clean_text

# Download required NLTK data
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)


def load_and_preprocess(filepath="fake_or_real_news.csv"):
    """Load dataset, preprocess, and return features + labels."""
    df = pd.read_csv(filepath)

    # Keep only required columns
    df = df[["title", "text", "label"]].copy()

    # Convert labels: REAL -> 0, FAKE -> 1
    df["label"] = df["label"].map({"REAL": 0, "FAKE": 1})

    # Combine title and text into a single content column
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

    # Drop rows with missing labels
    df.dropna(subset=["label"], inplace=True)

    # Clean content
    df["content"] = df["content"].apply(clean_text)

    return df["content"], df["label"]


def train_and_evaluate():
    """Train the model and print evaluation metrics."""
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess()

    # Train / test split (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # TF-IDF vectorisation
    print("Vectorising text with TF-IDF...")
    tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Logistic Regression model
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluation
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model and vectorizer
    print("\nSaving model and vectorizer...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    print("Saved: model.pkl, vectorizer.pkl")

    return model, tfidf


if __name__ == "__main__":
    train_and_evaluate()
