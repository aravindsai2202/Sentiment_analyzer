import streamlit as st
import pickle
import re
import os

# FIX 4: Use file path so model loads correctly from any folder
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Sentilytics AI",
    page_icon="🧠",
    layout="centered"
)

# ---------------------------------------------------
# GEN-Z TRENDY BACKGROUND UI
# ---------------------------------------------------
st.markdown(
    """
    <style>

    /* Main App Background */
    .stApp {
        background-image:
        linear-gradient(rgba(10,10,10,0.82), rgba(10,10,10,0.88)),
        url("https://images.unsplash.com/photo-1518770660439-4636190af475");

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Main Container */
    .block-container {
        background: rgba(20, 20, 20, 0.65);
        padding: 2.5rem;
        border-radius: 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 0 40px rgba(255, 0, 150, 0.25);
        margin-top: 2rem;
    }

    /* Heading */
    h1 {
        color: white !important;
        text-align: center;
        font-size: 3rem !important;
        font-weight: 800;
        letter-spacing: 1px;
    }

    h2, h3, h4, p, label {
        color: white !important;
    }

    /* Text Area */
    textarea {
        background-color: rgba(255,255,255,0.08) !important;
        color: white !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        font-size: 17px !important;
    }

    /* Predict Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ff0080, #7928ca);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 0.8rem;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s;
    }

    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 0 25px rgba(255,0,120,0.6);
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #ff0080, #7928ca);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# LOAD MODEL
# FIX 4: Load from same folder as this file (works anywhere)
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("🧠 Sentilytics AI")

st.markdown(
    """
    ### 🚀 Real-Time Sentiment Analysis System

    One of the most in-demand Machine Learning applications today —
    used in customer support, social media monitoring,
    stock market prediction, and product review analysis.

    Analyze any text instantly using AI-powered sentiment prediction.
    """
)

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------
user_input = st.text_area(
    "💬 Enter Any Text",
    placeholder="Type a review, tweet, comment, or opinion here...",
    height=180
)

# ---------------------------------------------------
# TEXT CLEANING
# FIX 1: Now matches EXACTLY what was done during training
# (removes stopwords and short words, just like the notebook)
# ---------------------------------------------------
stop_words = set(stopwords.words('english'))

# correct
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if st.button("✨ Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        cleaned = clean_text(user_input)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        # FIX 2: No more fake 85% — show real confidence or nothing
        try:
            probability = model.predict_proba(vector)[0]
            confidence = round(max(probability) * 100, 2)
            show_confidence = True
        except AttributeError:
            show_confidence = False

        # FIX 5: Removed "neutral" result — model only gives 0 or 1
        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")

        # Confidence Score
        if show_confidence:
            st.subheader("⚡ Confidence Score")
            st.progress(int(confidence))
            st.write(f"### {confidence}% Confidence")

# ---------------------------------------------------
# APPLICATIONS SECTION
# ---------------------------------------------------
st.markdown("---")

st.subheader("🌍 Real-World Applications")

st.markdown("""
✅ Customer Support Analysis  
✅ Social Media Monitoring  
✅ Product Review Analysis  
✅ Brand Reputation Tracking  
✅ Financial Market Sentiment  
✅ Opinion Mining Systems  
""")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.caption("Built using Machine Learning, NLP, TF-IDF & Streamlit 🚀")
