from pathlib import Path
import pandas as pd
import joblib
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils import clean_text, label_to_int

DATASET_REF = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
CSV_NAME = "IMDB Dataset.csv"
MODEL_PATH = "best_sentiment_model.joblib"
REPORT_PATH = "model_report.txt"


def download_dataset() -> Path:
    print("Downloading dataset from Kaggle...")
    dataset_path = Path(kagglehub.dataset_download(DATASET_REF))
    csv_path = dataset_path / CSV_NAME

    if not csv_path.exists():
        matches = list(dataset_path.rglob("*.csv"))
        if not matches:
            raise FileNotFoundError("No CSV file found inside downloaded Kaggle dataset.")
        csv_path = matches[0]

    return csv_path


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = {"review", "sentiment"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}. Found: {list(df.columns)}")

    df = df[["review", "sentiment"]].dropna().copy()
    df["clean_review"] = df["review"].astype(str).apply(clean_text)
    df = df[df["clean_review"].str.len() > 0].copy()
    df["label"] = df["sentiment"].apply(label_to_int)

    return df


def build_models():
    nb_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=30000, ngram_range=(1, 2))),
        ("clf", MultinomialNB()),
    ])

    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=30000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=2000)),
    ])

    return {
        "Naive Bayes": nb_pipeline,
        "Logistic Regression": lr_pipeline,
    }


def evaluate_model(name, model, x_train, x_test, y_train, y_test):
    print(f"\nTraining: {name}")
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=["Negative", "Positive"])

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(report)

    return {
        "name": name,
        "model": model,
        "accuracy": acc,
        "f1": f1,
        "report": report,
    }


def main():
    csv_path = download_dataset()
    df = load_and_prepare_data(csv_path)

    x = df["clean_review"]
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models()
    results = []

    for name, model in models.items():
        result = evaluate_model(name, model, x_train, x_test, y_train, y_test)
        results.append(result)

    best = max(results, key=lambda r: r["f1"])
    joblib.dump(best["model"], MODEL_PATH)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Sentiment Analysis Model Comparison\n")
        f.write("=" * 40 + "\n\n")
        for r in results:
            f.write(f"Model: {r['name']}\n")
            f.write(f"Accuracy: {r['accuracy']:.4f}\n")
            f.write(f"F1 Score: {r['f1']:.4f}\n")
            f.write(r["report"])
            f.write("\n" + "-" * 60 + "\n\n")
        f.write(f"Best Model: {best['name']}\n")

    print(f"\nBest model saved to: {MODEL_PATH}")
    print(f"Evaluation report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
