# Sentiment Analysis Tool (Project 2)

This project satisfies the requirements in your screenshot:

- Load labeled text data from Kaggle (IMDB 50K reviews)
- Clean and preprocess text
- Convert text to numeric features using TF-IDF
- Train Logistic Regression and Naive Bayes
- Evaluate with Accuracy and F1 Score
- Provide a CLI for predictions
- Provide a polished Streamlit web interface

## Dataset
Recommended Kaggle dataset:
- IMDB Dataset of 50K Movie Reviews: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

The Kaggle dataset provides 50,000 movie reviews with `review` and `sentiment` columns for binary classification. citeturn397672search0

## Project structure

- `train_model.py` -> downloads/loads data, trains models, saves best model
- `app.py` -> Streamlit web interface
- `cli.py` -> command-line prediction tool
- `requirements.txt` -> packages to install
- `utils.py` -> shared preprocessing utilities

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## Kaggle API setup

1. Go to Kaggle -> Account -> Create New API Token
2. Download `kaggle.json`
3. Put it here:

### Windows
```bash
C:\Users\YourName\.kaggle\kaggle.json
```

### Mac/Linux
```bash
~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Train the model

```bash
python train_model.py
```

This will:
- download the IMDB dataset from Kaggle
- preprocess the reviews
- train Naive Bayes and Logistic Regression
- compare both models
- save the best pipeline as `best_sentiment_model.joblib`

## Run CLI

```bash
python cli.py
```

## Run web app

```bash
streamlit run app.py
```

## Notes

- The web app is made to look clean and client-friendly.
- Logistic Regression usually performs better than Naive Bayes on TF-IDF text classification, but the script tests both.
