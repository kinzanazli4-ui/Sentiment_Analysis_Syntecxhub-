from pathlib import Path
import joblib
import streamlit as st
from utils import clean_text, int_to_label

st.set_page_config(page_title="Sentiment Analysis Studio", page_icon="💬", layout="wide")

MODEL_PATH = Path("best_sentiment_model.joblib")

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #111827 50%, #1e293b 100%);
        color: white;
    }
    .hero-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        padding: 28px;
        border-radius: 22px;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    .result-positive {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.45);
        padding: 20px;
        border-radius: 18px;
    }
    .result-negative {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.45);
        padding: 20px;
        border-radius: 18px;
    }
    .small-card {
        background: rgba(255,255,255,0.05);
        border-radius: 18px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

st.markdown("<div class='hero-card'>", unsafe_allow_html=True)
st.title("💬 Sentiment Analysis Studio")
st.subheader("Turn raw text into clear positive or negative sentiment insights")
st.write(
    "A clean and client-friendly NLP interface built with TF-IDF and machine learning. "
    "Use it for reviews, feedback, comments, and quick opinion analysis."
)
st.markdown("</div>", unsafe_allow_html=True)

st.write("")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='small-card'><h4>Fast Prediction</h4><p>Paste any review or feedback and get instant sentiment output.</p></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='small-card'><h4>Business Friendly</h4><p>Useful for client demos, product reviews, and brand monitoring mockups.</p></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='small-card'><h4>Model Backed</h4><p>Built using real labeled sentiment data and evaluated with accuracy and F1 score.</p></div>", unsafe_allow_html=True)

st.write("")

if model is None:
    st.warning("Model not found. Please run `python train_model.py` first.")
else:
    input_text = st.text_area(
        "Enter text to analyze",
        height=180,
        placeholder="Example: This product was surprisingly good and the experience felt smooth and reliable.",
    )

    btn_col1, btn_col2 = st.columns([1, 5])
    with btn_col1:
        analyze = st.button("Analyze")

    if analyze:
        cleaned = clean_text(input_text)

        if not cleaned:
            st.error("Please enter some valid text first.")
        else:
            pred = model.predict([cleaned])[0]
            probs = model.predict_proba([cleaned])[0]
            label = int_to_label(pred)
            confidence = float(max(probs) * 100)

            if label == "Positive":
                st.markdown(
                    f"<div class='result-positive'><h3>Prediction: {label}</h3><p>Confidence: {confidence:.2f}%</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='result-negative'><h3>Prediction: {label}</h3><p>Confidence: {confidence:.2f}%</p></div>",
                    unsafe_allow_html=True,
                )

            st.write("")
            st.caption("Model output is based on the trained TF-IDF + machine learning pipeline.")

st.write("")
st.markdown("### Demo texts")
example_col1, example_col2 = st.columns(2)
with example_col1:
    st.info("Positive example: I loved the experience, the interface felt premium and everything worked perfectly.")
with example_col2:
    st.info("Negative example: The service was frustrating, slow, and full of issues from start to finish.")
