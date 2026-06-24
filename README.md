# SMS Spam Classifier

An end-to-end Machine Learning web application that classifies traditional SMS messages as **Spam** or **Not Spam** using Natural Language Processing (NLP). Built with Python, Scikit-Learn, and Streamlit, and deployed live on Streamlit Cloud.

##  Live Demo
🔗 **[Click here to try the live app!](https://sms-spam-classifier-amwa6b53r9bpwekxdnsyzt.streamlit.app/)**

---

## Features
* **Real-time Prediction:** Type or paste any short message to immediately check if it's spam.
* **NLP Pipeline:** Cleans text by converting to lowercase, tokenizing, alphanumeric filtering, removing stopwords/punctuation, and applying Porter Stemming.
* **Experimented with vectorization techniques:** Used CountVectorizer and TF-IDF and compared the performances
* **Ensemble Modeling:** Ulitmately, leverages a `VotingClassifier` utilizing Multinomial Naive Bayes and other well performing classifiers (RFC, ETC, SVC), to optimize classification accuracy and reduce false positives.

---

## 📁 Repository Structure
```text
├── app.py                  # Streamlit web application script
├── model.pkl               # Trained VotingClassifier ML model
├── vectorizer.pkl          # Trained TF-IDF Vectorizer
├── requirements.txt        # Production dependencies for cloud deployment
└── README.md               # Project documentation
