import joblib
from pathlib import Path
from utils import clean_text, int_to_label

MODEL_PATH = Path("best_sentiment_model.joblib")


def main():
    if not MODEL_PATH.exists():
        print("Model file not found. First run: python train_model.py")
        return

    model = joblib.load(MODEL_PATH)
    print("Sentiment CLI is ready. Type 'exit' to quit.\n")

    while True:
        text = input("Enter text: ").strip()
        if text.lower() == "exit":
            print("Goodbye!")
            break

        cleaned = clean_text(text)
        if not cleaned:
            print("Please enter valid text.\n")
            continue

        pred = model.predict([cleaned])[0]
        probs = model.predict_proba([cleaned])[0]

        label = int_to_label(pred)
        confidence = max(probs) * 100

        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2f}%\n")


if __name__ == "__main__":
    main()
