import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt

# 🔧 Setup NLTK stopwords for Streamlit Cloud
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, "corpora/stopwords")):
    nltk.download("stopwords", download_dir=nltk_data_path)

stop_words = set(stopwords.words("english"))

# 📦 Load model and TF-IDF vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# 🧹 Text cleaning
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# 🌐 Streamlit UI
st.title('💬 Sentiment Analysis App')
st.write("Enter a production/movie review below to predict if it's **Positive** or **Negative**.")

input_text = st.text_area("✍️ Enter your review:", "")

if st.button('Predict Sentiment'):
    if input_text.strip() == "":
        st.warning('Please enter a review.')
    else:
        # Preprocess and predict
        cleaned = clean_text(input_text)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probas = model.predict_proba(vectorized)[0]

        confidence = probas[prediction] * 100
        label = "Positive" if prediction == 1 else "Negative"

        # ✅ Display result
        if prediction == 1:
            st.success(f"✅ Positive Review ({confidence:.2f}% confidence)")
        else:
            st.error(f"❌ Negative Review ({confidence:.2f}% confidence)")

        # 📊 Pie Chart
        st.subheader("📊 Sentiment Confidence Breakdown")
        fig, ax = plt.subplots()
        ax.pie(probas, labels=["Negative", "Positive"], autopct='%1.1f%%',
               colors=["#ff4b4b", "#00cc88"], startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

# 👣 Footer
st.markdown("---")
st.markdown("👨‍💻 Built with ❤️ by [Abhishek Sharma](https://www.linkedin.com/in/abhishekksharmma/)")
