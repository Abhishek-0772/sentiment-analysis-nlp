import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt

# NLTK data setup
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)
if not os.path.exists(nltk_data_path + "/corpora/stopwords"):
    nltk.download("stopwords", download_dir=nltk_data_path)

stop_words = set(stopwords.words("english"))

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# App layout
st.title('💬 Sentiment Analysis App')
st.write("Enter a product or movie review below to predict if it's **Positive** or **Negative**.")

# ✍️ Text input
input_text = st.text_area("✍️ Enter your review:", "")

# ✅ Sample reviews dropdown
with st.expander("📋 Try Sample IMDB-style Reviews"):
    examples = [
        "This movie was an emotional rollercoaster. Absolutely loved it!",
        "The plot was predictable and the acting was terrible.",
        "The movie was boring in its entirety, but that made a beautiful impact! Give it a chance.",
        "It's not the worst movie I've seen, but far from good.",
        "Amazing visuals and soundtrack. Mediocre story though.",
        "A must-watch. The direction and acting were top-notch.",
        "What a waste of time. I regret watching it.",
        "Decent storyline, but pacing was painfully slow.",
        "It touched my heart. Truly a beautiful film.",
        "The worst movie I’ve ever seen. Don’t bother."
    ]
    selected = st.selectbox("Choose a sample review:", [""] + examples)
    if selected:
        input_text = selected

# 🧠 Prediction
if st.button('Predict Sentiment'):
    if input_text.strip() == "":
        st.warning('Please enter a review.')
    else:
        cleaned = clean_text(input_text)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probas = model.predict_proba(vectorized)[0]

        confidence = probas[prediction] * 100
        label = "Positive" if prediction == 1 else "Negative"

        # 🎯 Display prediction
        if prediction == 1:
            st.success(f"✅ Positive Review ({confidence:.2f}% confidence)")
        else:
            st.error(f"❌ Negative Review ({confidence:.2f}% confidence)")

        # 📊 Pie chart
        st.subheader("📊 Sentiment Confidence Breakdown")
        fig, ax = plt.subplots()
        ax.pie(probas, labels=["Negative", "Positive"], autopct='%1.1f%%',
               colors=["#ff4b4b", "#00cc88"], startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("👨‍💻 Built with ❤️ by [Abhishek Sharma](https://www.linkedin.com/in/abhishekksharmma/)")
