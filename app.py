import streamlit as st
import pickle
import re
from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk

# Download stopwords
nltk.download("stopwords")

# ---------------- Load Models ----------------
with open("model/log_model.pkl", "rb") as f:
    log_model = pickle.load(f)

with open("model/nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("model/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("model/voting_model.pkl", "rb") as f:
    voting_model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---------------- Load Stopwords ----------------
def load_stopwords(path):
    with open(path, "r", encoding="utf-8") as f:
        return set([w.strip() for w in f.readlines() if w.strip()])

hindi_stop = load_stopwords(r"C:\Users\vinot\Projects\Fake News Detection\data\stopwords\hindi_stopwords.txt")
tamil_stop = load_stopwords(r"C:\Users\vinot\Projects\Fake News Detection\data\stopwords\Tamil-Stopwords.txt")
english_stop = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

# ---------------- Preprocessing ----------------
def multilingual_preprocess(text):
    try:
        lang = detect(text)
    except:
        lang = "en"

    text = re.sub(r"[^a-zA-Z\u0900-\u097F\u0B80-\u0BFF ]", " ", str(text))
    words = text.lower().split()

    if lang == "en":
        words = [stemmer.stem(w) for w in words if w not in english_stop]
    elif lang == "hi":
        words = [w for w in words if w not in hindi_stop]
    elif lang == "ta":
        words = [w for w in words if w not in tamil_stop]

    return " ".join(words), lang


# ---------------- Streamlit App ----------------
st.title("📰 Multilingual Fake News Detection App")
st.markdown("Supports **English 🇬🇧, Hindi 🇮🇳, Tamil 🇮🇳**")

st.markdown(
    """
    <div style="background-color:#FFF3CD;padding:10px;border-radius:5px;border:1px solid #FFEEBA;">
    ⚠️ <b>Disclaimer:</b> This system uses AI/ML models. Predictions may not always be accurate. 
    Please verify news from official/reliable sources before sharing.
    </div>
    """,
    unsafe_allow_html=True
)

# Text input
news_text = st.text_area("✍️ Paste the news content here:", height=200)

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some news content to analyze.")
    else:
        # Preprocess
        cleaned_text, lang = multilingual_preprocess(news_text)
        vectorized_text = vectorizer.transform([cleaned_text])

        # Voting Ensemble prediction
        result = voting_model.predict(vectorized_text)
        proba = voting_model.predict_proba(vectorized_text)[0]
        confidence = max(proba) * 100

        # Display
        st.write(f"🌐 Detected Language: **{lang.upper()}**")
        if result[0] == 1:
            st.success(f"✅ This news is **REAL**. (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"🚨 Warning: This news may be **FAKE**. (Confidence: {confidence:.2f}%)")

        # Individual model results
        lr_pred = log_model.predict(vectorized_text)[0]
        nb_pred = nb_model.predict(vectorized_text)[0]
        rf_pred = rf_model.predict(vectorized_text)[0]

        st.subheader("🔍 Individual Model Predictions")
        st.write("⚖️ Logistic Regression:", "REAL ✅" if lr_pred == 1 else "FAKE 🚨")
        st.write("🧠 Naive Bayes:", "REAL ✅" if nb_pred == 1 else "FAKE 🚨")
        st.write("🌲 Random tree:", "REAL ✅" if rf_pred == 1 else "FAKE 🚨")
        
        st.info("ℹ️ Note: This is an AI prediction, not absolute truth.")