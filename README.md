# SMS Spam Classifier

A machine learning web app that classifies SMS messages as **Spam** or **Not Spam**, built with scikit-learn and served through a Streamlit interface.

🔗 **Live demo:** [https://sms-spam-classifier-amwa6b53r9bpwekxdnsyzt.streamlit.app/](#)

## Overview

This project trains and compares several classical ML algorithms on the [SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) and ships a good performing model behind a simple, interactive Streamlit UI. Type in a message, hit **Predict**, and get an instant spam/not spam verdict.

## Features

- Full text-preprocessing pipeline (lowercasing, tokenization, stopword/punctuation removal, stemming)
- Exploratory data analysis with visualizations (class balance, message length distribution, most frequent words)
- Benchmarking of 10+ classifiers (Naive Bayes variants, Logistic Regression, SVM, Random Forest, XGBoost, etc.)
- Final model: a hard-voting ensemble (MultinomialNB + Random Forest + SVM + Extra Trees)
- Lightweight Streamlit front end for real-time predictions

## How It Works

1. **Clean the data** — drop unused columns, rename fields, remove duplicates.
2. **Preprocess text** — lowercase → tokenize → keep alphanumeric tokens → remove stopwords/punctuation → stem with Porter Stemmer.
3. **Vectorize** — convert cleaned text to numerical features with `TfidfVectorizer` (top 3,000 features).
4. **Train & compare models** — evaluate accuracy and precision across multiple classifiers (precision is prioritized due to class imbalance).
5. **Ensemble the winners** — combine the strongest models into a `VotingClassifier` for the final, deployed model.
6. **Serve predictions** — `app.py` loads the saved vectorizer + model and exposes a Streamlit text box for live predictions.

## Model Performance

The dataset is imbalanced (~87% ham / 13% spam), so **precision** is the primary metric — false positives (flagging a real message as spam) are worse than missing the occasional spam message.

| Model | Accuracy | Precision |
|---|---|---|
| Multinomial Naive Bayes | 0.959 | 1.000 |
| Random Forest | 0.972 | 1.000 |
| Extra Trees | 0.973 | 0.982 |
| SVM (sigmoid kernel) | 0.973 | 0.974 |
| XGBoost | 0.971 | 0.943 |
| **Voting Classifier (final model)** | **0.977** | **1.000** |

> Full benchmark results, including Logistic Regression, KNN, Decision Tree, AdaBoost, Gradient Boosting, and a soft-voting/stacking comparison, are available in `sms_classifier.ipynb`.

## Project Structure

```
.
├── app.py                  # Streamlit app for live predictions
├── sms_classifier.ipynb    # Data cleaning, EDA, preprocessing, model training & comparison
├── spam.csv                # Raw SMS Spam Collection dataset
├── vectorizer.pkl          # Fitted TfidfVectorizer
├── model.pkl                # Trained VotingClassifier
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
# Clone the repo
git clone https://github.com/anjalichenga/sms-spam-classifier.git
cd sms-spam-classifier

# create a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal (usually `http://localhost:8501`), type a message into the text box, and click **Predict**.

### Retrain the model (optional)

Open `sms_classifier.ipynb` in Jupyter to walk through the full data cleaning, EDA, and model training process, or to retrain on your own data. The notebook ends by pickling the fitted vectorizer and model as `vectorizer.pkl` and `model.pkl`.

## Tech Stack

- **Python**
- **scikit-learn** — TF-IDF vectorization, model training, voting ensemble
- **NLTK** — tokenization, stopword removal, stemming
- **pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualizations
- **Streamlit** — web app interface

## Future Improvements

- [ ] Pin exact package versions in `requirements.txt` for full reproducibility
- [ ] Replace stemming with lemmatization and compare results
- [ ] Try transformer-based embeddings (e.g., DistilBERT) for richer text representations
- [ ] Add unit tests for the preprocessing pipeline
- [ ] Deploy to Streamlit Community Cloud / Hugging Face Spaces and link the live demo above

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,572 SMS messages labeled as ham or spam, originally compiled for academic spam-filtering research.

## Acknowledgments

- UCI Machine Learning Repository for the SMS Spam Collection dataset
- scikit-learn and NLTK documentation
