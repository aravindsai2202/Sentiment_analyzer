# 🧠 Sentilytics AI — Real-Time Sentiment Analysis

A machine learning web app that analyzes the sentiment of any text (positive or negative) in real time.

Built with Python, NLP, TF-IDF, Logistic Regression, and Streamlit.

---

## 🚀 Live Demo

👉 [Click here to try the app](https://your-app-link.streamlit.app) ← replace this after deploying

---

## 📸 Screenshot

<!-- Add a screenshot of your app here after deploying -->

---

## 🛠️ How It Works

1. User types any text (review, tweet, comment)
2. Text is cleaned — lowercased, symbols removed, stopwords removed
3. TF-IDF converts text into numbers the model understands
4. Logistic Regression model predicts: Positive or Negative
5. Confidence score is shown

---

## 📂 Project Structure

```
sentilytics-ai/
├── app.py               ← Streamlit web app
├── model.pkl            ← Trained Logistic Regression model
├── vectorizer.pkl       ← TF-IDF vectorizer
├── requirements.txt     ← Python libraries needed
├── notebooks/
│   └── step1_data.ipynb ← Full training notebook
└── data/
    └── reviews.csv      ← Dataset (NLTK movie reviews)
```

---

## ⚙️ Run Locally

```bash
# 1. Clone this repo
git clone https://github.com/YOUR_USERNAME/sentilytics-ai.git
cd sentilytics-ai

# 2. Install libraries
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📊 Model Performance

| Model               | Test Accuracy |
|---------------------|--------------|
| Logistic Regression | ~85%         |
| Naive Bayes         | ~80%         |

---

## 🌍 Real-World Applications

- Customer Support Analysis
- Social Media Monitoring
- Product Review Analysis
- Brand Reputation Tracking
- Financial Market Sentiment

---

## 🧰 Tech Stack

- Python
- Scikit-learn (TF-IDF + Logistic Regression)
- NLTK (text cleaning)
- Streamlit (web app)
- Pandas, NumPy

---

Built with ❤️ using Machine Learning and NLP
