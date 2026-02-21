# 📰 Fake News Sentiment Analysis

**A Data-Driven Approach to Understanding Public Emotion**

A complete Machine Learning web application that detects fake news and analyses the emotional tone of news articles using **Python**, **Scikit-learn**, **TextBlob**, and **Streamlit**.

---

## 📁 Project Structure

```
fakeNews/
├── fake_or_real_news.csv   # Dataset (title, text, label)
├── model_training.py       # Preprocessing, training & model persistence
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── model.pkl               # Saved Logistic Regression model (generated)
├── vectorizer.pkl          # Saved TF-IDF vectorizer (generated)
└── README.md
```

---

## 🔧 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/shankar5455/fakeNews.git
cd fakeNews
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Use the full dataset
The repository includes a sample dataset (`fake_or_real_news.csv`).  
For best accuracy, replace it with the full [GossipCop / FakeNewsNet](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news) dataset from Kaggle, which has the same `title`, `text`, `label` columns.

### 4. Train the model
```bash
python model_training.py
```
This generates `model.pkl` and `vectorizer.pkl`.

### 5. Launch the web application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 How It Works

| Step | Description |
|------|-------------|
| **Dataset** | CSV with `title`, `text`, `label` (REAL / FAKE) columns |
| **Preprocessing** | Lowercase, strip punctuation / URLs / numbers, combine title + text |
| **Feature Engineering** | TF-IDF Vectorizer (`stop_words="english"`, `max_df=0.7`) |
| **Model** | Logistic Regression (binary classification) |
| **Evaluation** | Accuracy, Classification Report, Confusion Matrix |
| **Sentiment** | TextBlob polarity → Positive 😊 / Negative 😡 / Neutral 😐 |

---

## 🌐 Deployment

### Streamlit Cloud (free)
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set **Main file** to `app.py`.
4. Add a startup command or `packages.txt` if needed.  
   > **Note:** You must commit `model.pkl` and `vectorizer.pkl` (run training locally first and push the files), or train inside the app on startup.

### Render
1. Create a new **Web Service** on [render.com](https://render.com).
2. Set **Build Command**: `pip install -r requirements.txt && python model_training.py`
3. Set **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## 🚀 Future Improvements

- Support for additional classifiers (Random Forest, SVM, XGBoost)
- Named-entity recognition to highlight suspicious claims
- REST API endpoint for programmatic access
- Multi-language fake news detection
- Real-time news feed integration

---

## 👥 Team Roles (4 Members)

| Member | Role |
|--------|------|
| **Member 1** | Data collection, cleaning & EDA |
| **Member 2** | Feature engineering & model training |
| **Member 3** | Sentiment analysis & model evaluation |
| **Member 4** | Streamlit UI, deployment & documentation |