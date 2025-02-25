import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import speech_recognition as sr


# 🔧 Initialize NLP tools
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

# ✅ Define text preprocessing function with error handling
def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":  # Ensure valid text input
        return ""

    try:
        tokens = word_tokenize(text)
        filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]
        return " ".join(filtered_tokens)
    except Exception as e:
        st.error(f"Error during text preprocessing: {e}")
        return ""

# ✅ Function to analyze sentiment safely
def analyze_sentiment(text):
    processed_text = preprocess_text(text)
    
    if not processed_text:  # If preprocessing failed or text is empty
        return "", 0.0, "Neutral"

    sentiment_score = sia.polarity_scores(processed_text)["compound"]
    sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
    return processed_text, sentiment_score, sentiment_label

# 🎨 Streamlit App Title
st.title("📊 NLP-Based Sentiment Analysis App")

# 📂 File Upload Section
st.header("Upload a CSV or TXT file for Sentiment Analysis")
uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=["csv", "txt"])

if uploaded_file is not None:
    # 🔹 Handling CSV & TXT Files
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:  # Handling TXT Files
        content = uploaded_file.getvalue().decode("utf-8")
        df = pd.DataFrame({"Text": content.splitlines()})

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # 🔹 Automatically detect text columns
    text_columns = [col for col in df.columns if df[col].dtype == "object"]
    if text_columns:
        selected_col = st.selectbox("Select the text column for analysis:", text_columns)

        # Remove empty values before analysis
        df = df.dropna(subset=[selected_col])

        df["Processed_Text"], df["Sentiment_Score"], df["Sentiment_Label"] = zip(*df[selected_col].apply(analyze_sentiment))

        st.write("### Sentiment Analysis Results")
        st.dataframe(df[[selected_col, "Processed_Text", "Sentiment_Label"]])

        # 📊 Sentiment Distribution
        st.write("### Sentiment Distribution")
        fig, ax = plt.subplots()
        df["Sentiment_Label"].value_counts().plot(kind="bar", ax=ax, color=["green", "red", "gray"])
        st.pyplot(fig)
    else:
        st.warning("No valid text column found in the uploaded file!")

# ✍ *Real-time Text Sentiment Analysis*
st.header("📝 Real-time Text Sentiment Analysis")
user_text = st.text_area("Enter text for sentiment analysis:", key="text_input_area")

if user_text:
    processed_text, sentiment_score, sentiment_label = analyze_sentiment(user_text)
    st.write(f"Processed Text: {processed_text}")
    st.write(f"Sentiment Score: {sentiment_score}")
    st.write(f"Sentiment Label: {sentiment_label}")

# 🎤 *Real-time Speech Sentiment Analysis*
st.header("🎤 Real-time Speech Sentiment Analysis")

if st.button("Start Recording"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Speak now!")
        audio = recognizer.listen(source)

        try:
            speech_text = recognizer.recognize_google(audio)
            st.write(f"Recognized Speech: {speech_text}")

            processed_text, sentiment_score, sentiment_label = analyze_sentiment(speech_text)
            st.write(f"Processed Text: {processed_text}")
            st.write(f"Sentiment Score: {sentiment_score}")
            st.write(f"Sentiment Label: {sentiment_label}")

        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
