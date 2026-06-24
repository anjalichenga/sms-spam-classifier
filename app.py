import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
ps = PorterStemmer()

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
PUNCTUATION = set(string.punctuation)

def preprocess_text (text) :
    text = text.lower()
    text = nltk.word_tokenize(text)
    k = []

    for i in text:
        if i.isalnum():
            k.append(i)
    text = k[:]
    k.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            k.append(i)
    text = k[:]
    k.clear()

    for i in text:
        k.append(ps.stem(i))
    return " ".join(k)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS Classifier (Spam/Not Spam)")

input_sms = st.text_area("Enter a message", height = 150)
if st.button("Predict"):
    if input_sms.strip() != "":

        preprocessed_sms = preprocess_text(input_sms)

        vectorized_sms = tfidf.transform([preprocessed_sms]).toarray()

        result = model.predict(vectorized_sms)

        if result[0] == 1:
            st.error("Spam")
        else:
            st.success("Not Spam")
    else:
        st.warning("No message entered")



