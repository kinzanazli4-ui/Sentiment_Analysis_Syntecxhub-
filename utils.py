import re
from html import unescape


def clean_text(text: str) -> str:
    """Basic text cleaning for sentiment analysis."""
    if not isinstance(text, str):
        return ""

    text = unescape(text)
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def label_to_int(label: str) -> int:
    label = str(label).strip().lower()
    return 1 if label == "positive" else 0


def int_to_label(value: int) -> str:
    return "Positive" if int(value) == 1 else "Negative"
