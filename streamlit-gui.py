import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from manualSVM import clean_text, preprocess, vectorize_tfidf, LinearSVM

st.set_page_config(page_title="Text Emotion Classification (SVM)", layout="centered")

# Load the model
@st.cache_resource
def load_model():
    with open("svm_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    return model_data

model_data = load_model()
model = model_data['model']
word2idx = model_data['word2idx']
idf = model_data['idf']
label2idx = model_data['label2idx']
idx2label = model_data['idx2label']
ID_STOPWORDS = model_data['stopwords']
accuracy = model_data['accuracy']
precision = model_data['precision']
recall = model_data['recall']
f1 = model_data['f1']

def predict_emotion(text):
    cleaned = clean_text(text)
    tokens = preprocess(cleaned)
    vector = vectorize_tfidf(tokens, word2idx, idf)
    prediction = model.predict(np.array([vector]))[0]
    return idx2label[prediction]

st.title("Text Emotion Classification using Support Vector Machine")
st.divider()

input_text = st.text_area("Try it now!", height=150, placeholder="Enter a sentence in Bahasa Indonesia and the model will predict the emotion")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = predict_emotion(input_text)
        st.success(f"**Predicted Emotion:** {result}")

st.divider()
st.subheader("Model Performance Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Precision", f"{precision:.2%}")
col3.metric("Recall", f"{recall:.2%}")
col4.metric("F1 Score", f"{f1:.2%}")

st.divider()
st.subheader("Authors")

data = [
    ["Danishwara Pracheta", "2308561050", "@dash4k"],
    ["Maliqy Numurti", "2308561068", "@Maliqytritata"],
    ["Krisna Udayana", "2308561122", "@KrisnaUdayana"],
    ["Dewa Sutha", "2308561137", "@DewaMahattama"]
]

# Create a 4-row, 3-column layout
for row in data:
    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            st.write(row[i])