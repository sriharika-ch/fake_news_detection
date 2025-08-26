import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection")
st.write("Enter a news article below to check if it's **REAL** or **FAKE**.")

# CSV file path
CSV_FILE = "fake_or_real_news.csv"
MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

# Step 1: Load dataset
if not os.path.exists(CSV_FILE):
    st.error(f"CSV file not found: {CSV_FILE}. Please check the path!")
else:
    st.text("Loading model... Please wait a few seconds if running first time.")

    # Step 2: Train model only if files do not exist
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        df = pd.read_csv(CSV_FILE)
        x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=7)
        accuracy = accuracy_score(y_test, model.predict(vectorizer.transform(x_test)))
    else:
        df = pd.read_csv(CSV_FILE)
        x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=7)

        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        tfidf_train = vectorizer.fit_transform(x_train)
        tfidf_test = vectorizer.transform(x_test)

        model = PassiveAggressiveClassifier(max_iter=100)
        model.fit(tfidf_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(tfidf_test))

        # Save model and vectorizer for future runs
        joblib.dump(model, MODEL_FILE)
        joblib.dump(vectorizer, VECTORIZER_FILE)

    # Step 3: User input
    user_input = st.text_area("🧾 Enter News Text:", height=250)

    if st.button("Predict"):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text.")
        else:
            input_vec = vectorizer.transform([user_input])
            prediction = model.predict(input_vec)[0]
            confidence = model.decision_function(input_vec)[0]

            st.subheader("Prediction:")
            if prediction == "FAKE":
                st.error(f"🚫 This news is likely **FAKE**.\nConfidence score: {confidence:.2f}")
            else:
                st.success(f"✅ This news appears to be **REAL**.\nConfidence score: {confidence:.2f}")

    # Step 4: Sidebar with model info
    st.sidebar.title("ℹ️ Model Info")
    st.sidebar.write(f"**Model Accuracy:** {accuracy * 100:.2f}%")
    st.sidebar.markdown(
        "Model: `PassiveAggressiveClassifier`<br>Vectorizer: `TfidfVectorizer`",
        unsafe_allow_html=True
    )
