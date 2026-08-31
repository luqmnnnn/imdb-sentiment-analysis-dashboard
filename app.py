import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

DATA_PATH_DEFAULT = "IMDB Dataset.csv"
EMOTION_LABELS_PATH = "emotion_labels.csv"
DISTILBERT_CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"

# ── Cafe design tokens ──────────────────────────────────────
PRIMARY   = "#5D4432"   # coffee brown
SECONDARY = "#E9E3DD"   # warm cream
SURFACE   = "#F9F7F5"   # off-white warm
TEXT      = "#3E2B1E"   # dark espresso
SUCCESS   = "#16A34A"
WARNING   = "#D97706"
DANGER    = "#DC2626"
MUTED     = "#9C8475"   # warm taupe

st.set_page_config(
    page_title="Movie Review Sentiment — Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* =================================================================
       CAFE DESIGN SYSTEM — Poppins font + warm brown/cream palette
       Tokens: primary=#5D4432 secondary=#E9E3DD surface=#F9F7F5
                text=#3E2B1E  success=#16A34A warning=#D97706
    ================================================================= */

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Poppins', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(-45deg, #1a0d08, #3a1f12, #5D4432, #8B6040, #C4A882, #8B6040, #3a1f12);
        background-size: 400% 400%;
        animation: cafeWarm 22s ease infinite;
    }
    @keyframes cafeWarm {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .block-container {
        background: rgba(249, 247, 245, 0.97) !important;
        border-radius: 20px !important;
        padding: 2.4rem 3rem 3.2rem !important;
        margin-top: 1rem !important;
        box-shadow: 0 24px 64px rgba(61,27,14,0.30) !important;
        backdrop-filter: blur(14px);
        border-top: 3px solid #5D4432;
    }

    .block-container > div,
    .stMarkdown p, .stMarkdown li, .stMarkdown ol,
    .stMarkdown ul, .stMarkdown strong, .stMarkdown em,
    .stCaption {
        color: #3E2B1E !important;
        font-family: 'Poppins', sans-serif !important;
    }

    .block-container h1 {
        color: #5D4432 !important;
        font-weight: 700 !important;
        font-family: 'Poppins', sans-serif !important;
    }
    .block-container h2,
    .block-container h3 {
        color: #5D4432 !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
        border-left: 4px solid #D97706;
        padding-left: 0.65rem;
        margin-top: 1.8rem;
    }
    .block-container h4 {
        color: #5D4432 !important;
        font-weight: 600 !important;
    }

    .stSelectbox > label,
    .stMultiSelect > label,
    .stSlider > label,
    .stTextInput > label,
    .stTextArea > label,
    .stRadio > label,
    .stCheckbox > label,
    .stNumberInput > label {
        color: #3E2B1E !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    [data-baseweb="select"] {
        background-color: #ffffff !important;
        border-radius: 8px !important;
    }
    [data-baseweb="select"] * {
        color: #3E2B1E !important;
        background-color: transparent !important;
        font-family: 'Poppins', sans-serif !important;
    }
    [data-baseweb="select"] [data-baseweb="tag"] {
        background-color: #E9E3DD !important;
        color: #5D4432 !important;
        border-radius: 20px !important;
    }
    [data-baseweb="popover"],
    [data-baseweb="menu"] {
        background-color: #ffffff !important;
        border: 1px solid #E9E3DD !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 24px rgba(93,68,50,0.18) !important;
    }
    [data-baseweb="option"] {
        color: #3E2B1E !important;
        background-color: #ffffff !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 0.9rem !important;
    }
    [data-baseweb="option"]:hover,
    [data-baseweb="option"][aria-selected="true"] {
        background-color: #E9E3DD !important;
        color: #5D4432 !important;
    }

    [data-baseweb="input"] input {
        background-color: #ffffff !important;
        color: #3E2B1E !important;
        border: 1.5px solid #C4A882 !important;
        border-radius: 8px !important;
        font-family: 'Poppins', sans-serif !important;
    }

    /* --- Text area (e.g. "Write a movie review") — dark background, white text --- */
    [data-baseweb="textarea"] textarea,
    .stTextArea textarea {
        background-color: #3E2B1E !important;
        color: #F9F7F5 !important;
        border: 1.5px solid #C4A882 !important;
        border-radius: 8px !important;
        font-family: 'Poppins', sans-serif !important;
        -webkit-text-fill-color: #F9F7F5 !important;
        text-shadow: none !important;
    }
    [data-baseweb="textarea"] textarea::placeholder,
    .stTextArea textarea::placeholder {
        color: #C4A882 !important;
        opacity: 1 !important;
    }

    [data-baseweb="input"] input:focus,
    [data-baseweb="textarea"] textarea:focus,
    .stTextArea textarea:focus {
        border-color: #5D4432 !important;
        box-shadow: 0 0 0 3px rgba(93,68,50,0.15) !important;
    }
    [data-baseweb="input"] input::placeholder {
        color: #9C8475 !important;
    }

    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"],
    .stSlider span {
        color: #3E2B1E !important;
        font-family: 'Poppins', sans-serif !important;
    }

    /* --- Slider floating value bubble + spacing (prevents overlap) --- */
    .stSlider {
        padding-top: 1.3rem !important;
        padding-bottom: 0.4rem !important;
    }
    .stSlider [data-testid="stSliderThumbValue"] {
        color: #5D4432 !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
    }
    .stSlider [data-testid="stTickBar"],
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        margin-top: 0.3rem !important;
    }

    /* --- Caption text (uses stCaptionContainer testid, not a .stCaption class) --- */
    [data-testid="stCaptionContainer"] p {
        color: #3E2B1E !important;
        font-family: 'Poppins', sans-serif !important;
    }

    /* =================================================================
       SIDEBAR — dark espresso + cream text
    ================================================================= */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0d08 0%, #2C1810 55%, #3a1f12 100%) !important;
        border-right: 1px solid rgba(196,168,130,0.20);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        font-family: 'Poppins', sans-serif !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #C4A882 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.1px;
        transition: color 0.15s;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #F9F7F5 !important;
    }
    section[data-testid="stSidebar"] .stRadio [data-checked="true"] + div p {
        color: #F9F7F5 !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] .stRadio label p {
        color: inherit !important;
        font-weight: inherit !important;
    }
    section[data-testid="stSidebar"] .stTextInput label p,
    section[data-testid="stSidebar"] .stSlider label p {
        color: #F9F7F5 !important;
    }
    section[data-testid="stSidebar"] .stTextInput > label,
    section[data-testid="stSidebar"] .stSlider > label {
        color: #F9F7F5 !important;
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"],
    section[data-testid="stSidebar"] .stSlider span {
        color: #E9E3DD !important;
    }
    section[data-testid="stSidebar"] .stSlider [data-testid="stSliderThumbValue"] {
        color: #F9F7F5 !important;
        font-weight: 700 !important;
    }
    /* Caption text was dark-on-dark and unreadable inside the sidebar.
       Streamlit renders captions via [data-testid="stCaptionContainer"], not a .stCaption class. */
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
    section[data-testid="stSidebar"] small {
        color: #C4A882 !important;
        opacity: 0.95 !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="input"] input {
        background-color: rgba(255,255,255,0.06) !important;
        color: #E9E3DD !important;
        border: 1px solid rgba(196,168,130,0.30) !important;
    }
    section[data-testid="stSidebar"] strong {
        color: #D97706 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #C4A882 !important;
        border: none !important;
        padding: 0 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(196,168,130,0.20) !important;
        margin: 0.8rem 0 !important;
    }
    section[data-testid="stSidebar"] .stAlert * {
        color: #ffb3b3 !important;
    }

    /* =================================================================
       PAGE HEADER STRIP
    ================================================================= */
    .page-header {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1.5px solid #E9E3DD;
    }
    .page-header-bar {
        width: 5px;
        min-height: 56px;
        border-radius: 3px;
        background: linear-gradient(180deg, #5D4432, #D97706);
        flex-shrink: 0;
        margin-top: 2px;
    }
    .page-header h1 {
        margin: 0 !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #5D4432 !important;
        font-family: 'Poppins', sans-serif !important;
        border: none !important;
        padding: 0 !important;
        line-height: 1.2;
        letter-spacing: -0.3px;
    }
    .page-header .ph-sub {
        font-size: 0.85rem;
        color: #9C8475 !important;
        margin-top: 0.3rem;
        font-weight: 400;
        font-family: 'Poppins', sans-serif !important;
    }

    /* =================================================================
       METRIC CARDS
    ================================================================= */
    .metric-card {
        background: linear-gradient(135deg, #5D4432 0%, #7A5C42 60%, #D97706 100%);
        padding: 1.5rem 1.2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(93,68,50,0.35);
        transition: transform 0.22s ease, box-shadow 0.22s ease;
        font-family: 'Poppins', sans-serif;
    }
    .metric-card:hover {
        transform: translateY(-6px) scale(1.03);
        box-shadow: 0 16px 40px rgba(93,68,50,0.45);
    }
    .metric-card .num {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #F9F7F5 !important;
        line-height: 1.2;
        font-family: 'Poppins', sans-serif !important;
    }
    .metric-card .lbl {
        font-size: 0.75rem !important;
        color: #E9E3DD !important;
        margin-top: 0.35rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-family: 'Poppins', sans-serif !important;
    }

    /* =================================================================
       BUTTONS
    ================================================================= */
    .stButton > button {
        background: #5D4432 !important;
        color: #F9F7F5 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.6rem !important;
        font-weight: 600 !important;
        font-size: 0.91rem !important;
        font-family: 'Poppins', sans-serif !important;
        letter-spacing: 0.2px;
        transition: all 0.20s ease !important;
        box-shadow: 0 3px 10px rgba(93,68,50,0.28) !important;
    }
    .stButton > button:hover {
        background: #D97706 !important;
        color: #1a0d08 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(217,119,6,0.40) !important;
    }

    /* =================================================================
       TECH PILLS
    ================================================================= */
    .pill {
        display: inline-block;
        background: #E9E3DD;
        color: #5D4432 !important;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.80rem;
        margin: 4px;
        font-weight: 600;
        border: 1.5px solid #C4A882;
        font-family: 'Poppins', sans-serif;
        letter-spacing: 0.1px;
    }

    .stAlert p, .stAlert div, .stAlert span {
        color: #3E2B1E !important;
        font-family: 'Poppins', sans-serif !important;
    }

    .stDataFrame * {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# PAGE HEADER HELPER
# ─────────────────────────────────────────────────────────────
def page_header(title, subtitle=""):
    sub_html = f'<div class="ph-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-header-bar"></div>
            <div>
                <h1>{title}</h1>
                {sub_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# MATPLOTLIB STYLE — cafe warm palette
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "#F9F7F5",
    "figure.facecolor": "#F9F7F5",
    "axes.edgecolor": "#C4A882",
    "axes.labelcolor": "#3E2B1E",
    "xtick.color": "#3E2B1E",
    "ytick.color": "#3E2B1E",
    "text.color": "#3E2B1E",
    "grid.color": "#E9E3DD",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

CAFE_BAR    = "#5D4432"
CAFE_LINE   = "#D97706"
CAFE_ACCENT = "#8B6040"


def ensure_nltk():
    import nltk
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(pkg, quiet=True)


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_data
def preprocess_reviews(text_series):
    """Same pipeline as the notebook: strip <br/>, lowercase, drop non-letters,
    remove stopwords, lemmatize."""
    ensure_nltk()
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def clean(text):
        text = str(text).replace("<br />", " ")
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
        return " ".join(tokens)

    return text_series.apply(clean)


@st.cache_data
def get_sample(path, n, seed=42):
    df = load_data(path)
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    df["label"] = (df["sentiment"] == "positive").astype(int)
    df["clean_review"] = preprocess_reviews(df["review"])
    return df


@st.cache_resource(show_spinner="Training TF-IDF + classical models on the sample...")
def train_models(path, n, seed=42):
    df = get_sample(path, n, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"], df["label"], test_size=0.2, random_state=seed, stratify=df["label"]
    )
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vec.fit_transform(X_train)
    X_test_tfidf  = vec.transform(X_test)

    models = {
        "Random Forest":   RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "KNN":             KNeighborsClassifier(n_neighbors=5),
        "SVM (LinearSVC)": LinearSVC(random_state=seed),
        "AdaBoost":        AdaBoostClassifier(n_estimators=100, random_state=seed),
    }
    fitted, preds, scores = {}, {}, {}
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        fitted[name] = model
        y_pred = model.predict(X_test_tfidf)
        preds[name] = y_pred
        scores[name] = accuracy_score(y_test, y_pred)

    return {
        "vectorizer": vec, "models": fitted, "preds": preds, "scores": scores,
        "X_test": X_test, "y_test": y_test, "X_test_tfidf": X_test_tfidf,
    }


@st.cache_resource(show_spinner="Loading DistilBERT (first run downloads the model)...")
def load_distilbert():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=DISTILBERT_CHECKPOINT, device=-1)


@st.cache_resource(show_spinner="Training TF-IDF + SVM on the emotion subset...")
def train_emotion_model(imdb_path, emotion_path, seed=42):
    df = load_data(imdb_path)
    em = pd.read_csv(emotion_path)
    em = em.rename(columns={em.columns[0]: "row_idx"})
    merged = em.merge(df, left_on="row_idx", right_index=True)
    merged["clean_review"] = preprocess_reviews(merged["review"])

    X_train, X_test, y_train, y_test = train_test_split(
        merged["clean_review"], merged["emotion"], test_size=0.2, random_state=seed, stratify=merged["emotion"]
    )
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vec.fit_transform(X_train)
    X_test_tfidf  = vec.transform(X_test)

    clf = LinearSVC(random_state=seed)
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    baseline = y_test.value_counts(normalize=True).max()
    return {
        "merged": merged, "vec": vec, "clf": clf,
        "y_test": y_test, "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "baseline": baseline,
    }


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding:1.2rem 0.4rem 0.5rem;">
            <div style="font-size:0.68rem;letter-spacing:2.5px;text-transform:uppercase;
                        color:#9C8475;font-weight:600;margin-bottom:0.4rem;
                        font-family:'Poppins',sans-serif;">
                Analytics Dashboard
            </div>
            <div style="font-size:1.18rem;font-weight:800;color:#E9E3DD;line-height:1.3;
                        font-family:'Poppins',sans-serif;">
                Movie Review<br>
                <span style="color:#D97706;">Sentiment Analysis</span>
            </div>
        </div>
        <hr style="border-color:rgba(196,168,130,0.18);margin:0 0 0.6rem 0;">
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        [
            "Home",
            "Dataset Overview",
            "Preprocessing & TF-IDF",
            "Classification Analysis",
            "DistilBERT Benchmark",
            "Emotion Classification",
            "Try It Yourself",
            "Conclusion",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        """<hr style="border-color:rgba(196,168,130,0.18);margin:0.8rem 0 0.6rem;">""",
        unsafe_allow_html=True,
    )

    path = st.text_input("Dataset path", value=DATA_PATH_DEFAULT)
    if not os.path.exists(path):
        st.error("Dataset file not found.")

    sample_size = st.slider("Training sample size", 1000, 8000, 3000, 500)
    st.caption("Classical models retrain on this many reviews. Smaller = faster.")

    st.markdown(
        """
        <hr style="border-color:rgba(196,168,130,0.18);margin:0.8rem 0 0.6rem;">
        <div style="font-size:0.84rem;color:#C4A882;line-height:1.8;
                    font-family:'Poppins',sans-serif;">
            <strong>Dataset</strong><br><span style="color:#E9E3DD;">IMDB 50K Movie Reviews</span><br>
            <strong>Models</strong><br><span style="color:#E9E3DD;">RF, KNN, SVM, AdaBoost, DistilBERT</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
data_loaded = os.path.exists(path)
if data_loaded:
    df_raw = load_data(path)


# ============================================================
# MODULE 1 — HOME
# ============================================================
if page == "Home":
    page_header(
        "Movie Review Sentiment Analysis",
        "TF-IDF classical models benchmarked against a pretrained DistilBERT",
    )

    st.markdown("### About this Dashboard")
    st.write(
        "This dashboard turns the accompanying notebook into an interactive tool. It trains Random Forest, "
        "KNN, SVM and AdaBoost on TF-IDF vectors of IMDB movie reviews, benchmarks them against a pretrained "
        "DistilBERT running zero-shot, and extends the task to seven-way emotion classification. "
        "A live prediction page lets you type your own review and see what each model thinks."
    )

    st.markdown("#### Methods & Techniques")
    st.markdown(
        '<span class="pill">TF-IDF</span>'
        '<span class="pill">Random Forest</span>'
        '<span class="pill">KNN</span>'
        '<span class="pill">SVM</span>'
        '<span class="pill">AdaBoost</span>'
        '<span class="pill">DistilBERT</span>'
        '<span class="pill">ROC-AUC</span>'
        '<span class="pill">Cross-Validation</span>'
        '<span class="pill">Emotion Classification</span>',
        unsafe_allow_html=True,
    )

    if data_loaded:
        st.markdown("#### Quick Stats")
        pos_pct = (df_raw["sentiment"] == "positive").mean() * 100
        avg_len = df_raw["review"].str.len().mean()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="num">{df_raw.shape[0]:,}</div><div class="lbl">Reviews</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="num">{pos_pct:.1f}%</div><div class="lbl">Positive</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="num">{avg_len:,.0f}</div><div class="lbl">Avg Chars/Review</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="num">7</div><div class="lbl">Emotion Classes</div></div>', unsafe_allow_html=True)
    else:
        st.error("Place 'IMDB Dataset.csv' in the same folder as app.py")


# ============================================================
# MODULE 2 — DATASET OVERVIEW
# ============================================================
elif page == "Dataset Overview":
    page_header("Dataset Overview", "Raw data, class balance, and review length")

    if not data_loaded:
        st.error("Dataset not found. Place the CSV next to app.py.")
        st.stop()

    st.markdown("### Raw Dataset Preview")
    n_rows = st.slider("Rows to preview", 5, 50, 10)
    st.dataframe(df_raw.head(n_rows))

    st.markdown("### Dataset Shape")
    st.write(f"Rows: **{df_raw.shape[0]:,}** &nbsp;|&nbsp; Columns: **{df_raw.shape[1]}**")

    st.markdown("### Class Balance")
    counts = df_raw["sentiment"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=[SUCCESS, DANGER])
    ax.set_ylabel("Count")
    ax.set_title("Positive vs Negative Reviews")
    st.pyplot(fig)

    st.markdown("### Review Length Distribution")
    lengths = df_raw["review"].str.len()
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.hist(lengths, bins=40, color=CAFE_BAR, edgecolor="#F9F7F5", linewidth=0.4)
    ax2.set_xlabel("Characters per review")
    ax2.set_ylabel("Count")
    ax2.set_title("Review Length Distribution")
    ax2.grid(True)
    st.pyplot(fig2)


# ============================================================
# MODULE 3 — PREPROCESSING & TF-IDF
# ============================================================
elif page == "Preprocessing & TF-IDF":
    page_header("Preprocessing & TF-IDF", "Text cleaning pipeline and top TF-IDF terms")

    if not data_loaded:
        st.error("Dataset not found.")
        st.stop()

    st.markdown("### Cleaning Pipeline")
    st.write(
        "Each review has its `<br />` line breaks stripped, gets lowercased, has non-letter characters "
        "removed, drops stopwords, and gets lemmatized so \"loved\"/\"loving\"/\"loves\" collapse into one token."
    )

    idx = st.slider("Preview row", 0, min(len(df_raw), 200) - 1, 0)
    raw_text = df_raw["review"].iloc[idx]
    cleaned_preview = preprocess_reviews(pd.Series([raw_text])).iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before**")
        st.info(raw_text[:600] + ("..." if len(raw_text) > 600 else ""))
    with col2:
        st.markdown("**After**")
        st.info(cleaned_preview[:600] + ("..." if len(cleaned_preview) > 600 else ""))

    st.markdown("### Top TF-IDF Terms (current sample)")
    df_sample = get_sample(path, sample_size)
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    matrix = vec.fit_transform(df_sample["clean_review"])
    scores = matrix.toarray().sum(axis=0)
    top_df = (pd.DataFrame({"Term": vec.get_feature_names_out(), "Score": scores})
              .sort_values("Score", ascending=False).head(15))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_df["Term"][::-1], top_df["Score"][::-1], color=CAFE_ACCENT, edgecolor="#F9F7F5")
    ax.set_xlabel("Summed TF-IDF Score")
    ax.set_title(f"Top 15 Terms Across {sample_size:,} Reviews")
    st.pyplot(fig)

    st.markdown("### Word Clouds")
    wc_col1, wc_col2 = st.columns(2)
    with wc_col1:
        pos_text = " ".join(df_sample[df_sample["label"] == 1]["clean_review"])
        wc_pos = WordCloud(width=500, height=350, background_color="#F9F7F5", colormap="Greens").generate(pos_text)
        fig_pos, ax_pos = plt.subplots(figsize=(6, 4))
        ax_pos.imshow(wc_pos)
        ax_pos.axis("off")
        ax_pos.set_title("Positive Reviews")
        st.pyplot(fig_pos)
    with wc_col2:
        neg_text = " ".join(df_sample[df_sample["label"] == 0]["clean_review"])
        wc_neg = WordCloud(width=500, height=350, background_color="#F9F7F5", colormap="Reds").generate(neg_text)
        fig_neg, ax_neg = plt.subplots(figsize=(6, 4))
        ax_neg.imshow(wc_neg)
        ax_neg.axis("off")
        ax_neg.set_title("Negative Reviews")
        st.pyplot(fig_neg)


# ============================================================
# MODULE 4 — CLASSIFICATION ANALYSIS
# ============================================================
elif page == "Classification Analysis":
    page_header("Classification Analysis", "Random Forest, KNN, SVM and AdaBoost on TF-IDF vectors")

    if not data_loaded:
        st.error("Dataset not found.")
        st.stop()

    bundle = train_models(path, sample_size)
    scores, preds, y_test = bundle["scores"], bundle["preds"], bundle["y_test"]

    st.markdown("### Accuracy Comparison")
    cols = st.columns(len(scores))
    best_model = max(scores, key=scores.get)
    for col, (name, acc) in zip(cols, scores.items()):
        star = " ★" if name == best_model else ""
        col.markdown(
            f'<div class="metric-card"><div class="num">{acc:.2%}</div>'
            f'<div class="lbl">{name}{star}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Confusion Matrix")
    model_name = st.selectbox("Select model", list(scores.keys()))
    cm = confusion_matrix(y_test, preds[model_name])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", ax=ax, linewidths=0.5,
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    st.pyplot(fig)

    st.markdown("### ROC-AUC Curves")
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    for name, model in bundle["models"].items():
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(bundle["X_test_tfidf"])[:, 1]
        else:
            score = model.decision_function(bundle["X_test_tfidf"])
        fpr, tpr, _ = roc_curve(y_test, score)
        auc = roc_auc_score(y_test, score)
        ax2.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC = 0.500)")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curves")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    st.markdown("### 5-Fold Cross-Validation")
    if st.button("Run Cross-Validation"):
        cv_rows = []
        for name, model in bundle["models"].items():
            cv_scores = cross_val_score(model, bundle["X_test_tfidf"], y_test, cv=5)
            cv_rows.append({"Model": name, "Mean Accuracy": cv_scores.mean(), "Std": cv_scores.std()})
        cv_df = pd.DataFrame(cv_rows)
        st.dataframe(cv_df)
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.bar(cv_df["Model"], cv_df["Mean Accuracy"], yerr=cv_df["Std"],
                color=CAFE_BAR, edgecolor="#F9F7F5", capsize=5)
        ax3.set_ylabel("Accuracy")
        ax3.set_title("5-Fold CV Accuracy (mean +/- std)")
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig3)

    if model_name == "SVM (LinearSVC)":
        st.markdown("### Top SVM Coefficients")
        clf = bundle["models"]["SVM (LinearSVC)"]
        vec = bundle["vectorizer"]
        coefs = pd.Series(clf.coef_[0], index=vec.get_feature_names_out())
        top_pos = coefs.sort_values(ascending=False).head(10)
        top_neg = coefs.sort_values().head(10)
        c1, c2 = st.columns(2)
        with c1:
            fig4, ax4 = plt.subplots(figsize=(5, 5))
            ax4.barh(top_pos.index[::-1], top_pos.values[::-1], color=SUCCESS)
            ax4.set_title("Pushes Toward Positive")
            st.pyplot(fig4)
        with c2:
            fig5, ax5 = plt.subplots(figsize=(5, 5))
            ax5.barh(top_neg.index[::-1], top_neg.values[::-1], color=DANGER)
            ax5.set_title("Pushes Toward Negative")
            st.pyplot(fig5)


# ============================================================
# MODULE 5 — DISTILBERT BENCHMARK
# ============================================================
elif page == "DistilBERT Benchmark":
    page_header("DistilBERT Benchmark", "Pretrained transformer, zero training, run on a small sample")

    if not data_loaded:
        st.error("Dataset not found.")
        st.stop()

    st.write(
        "DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) already learned sentiment before this "
        "app ever ran, so it gets zero training here, just inference. Running it on the full test set is slow "
        "on free hosting, so pick a small sample below."
    )

    n_bert = st.slider("Reviews to score with DistilBERT", 10, 100, 30, 10)

    if st.button("Run DistilBERT"):
        clf_pipeline = load_distilbert()
        sample = df_raw.sample(n=n_bert, random_state=42).reset_index(drop=True)
        with st.spinner(f"Scoring {n_bert} reviews..."):
            results = clf_pipeline(sample["review"].str.slice(0, 512).tolist(), truncation=True)
        sample["predicted"] = ["positive" if r["label"] == "POSITIVE" else "negative" for r in results]
        sample["confidence"] = [r["score"] for r in results]
        sample["correct"] = sample["predicted"] == sample["sentiment"]

        acc = sample["correct"].mean()
        st.markdown("### DistilBERT Accuracy")
        st.markdown(
            f'<div class="metric-card" style="max-width:260px">'
            f'<div class="num">{acc:.2%}</div>'
            f'<div class="lbl">Accuracy on {n_bert} reviews</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Sample Predictions")
        st.dataframe(sample[["review", "sentiment", "predicted", "confidence", "correct"]].head(20))

        st.markdown("### Compare Against Classical Models")
        bundle = train_models(path, sample_size)
        compare_df = pd.DataFrame(
            list(bundle["scores"].items()) + [("DistilBERT (zero-shot)", acc)],
            columns=["Model", "Accuracy"],
        ).sort_values("Accuracy", ascending=False)
        colors = ["#2563EB" if m.startswith("DistilBERT") else CAFE_BAR for m in compare_df["Model"]]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(compare_df["Model"], compare_df["Accuracy"], color=colors, edgecolor="#F9F7F5")
        ax.set_ylabel("Accuracy")
        ax.set_title("DistilBERT vs Classical Models")
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig)
        st.caption("Classical model accuracy is on their own held-out test split, not the DistilBERT sample above, so treat this as a rough comparison.")


# ============================================================
# MODULE 6 — EMOTION CLASSIFICATION
# ============================================================
elif page == "Emotion Classification":
    page_header("Emotion Classification", "Seven-way emotion labels instead of two polarities")

    if not data_loaded:
        st.error("Dataset not found.")
        st.stop()
    if not os.path.exists(EMOTION_LABELS_PATH):
        st.error(f"'{EMOTION_LABELS_PATH}' not found. This page reuses the notebook's cached emotion labels.")
        st.stop()

    st.write(
        "IMDB has no emotion annotation, so labels here come from `j-hartmann/emotion-english-distilroberta-base`, "
        "cached in `emotion_labels.csv`. This page trains a TF-IDF + SVM classifier on those silver labels, so it "
        "measures how well a bag-of-words model reproduces a transformer's emotion judgments, not human ground truth."
    )

    bundle = train_emotion_model(path, EMOTION_LABELS_PATH)

    st.markdown("### Class Distribution")
    dist = bundle["merged"]["emotion"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(dist.index, dist.values, color=CAFE_ACCENT, edgecolor="#F9F7F5")
    ax.set_ylabel("Count")
    ax.set_title("Emotion Label Distribution")
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig)

    st.markdown("### Model Performance")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="num">{bundle["accuracy"]:.2%}</div><div class="lbl">Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="num">{bundle["macro_f1"]:.2%}</div><div class="lbl">Macro F1</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="num">{bundle["baseline"]:.2%}</div><div class="lbl">Majority Baseline</div></div>', unsafe_allow_html=True)
    st.caption("Classes are imbalanced (disgust/neutral/joy dominate), so macro F1 and the majority baseline matter more than raw accuracy here.")

    st.markdown("### Confusion Matrix")
    labels = sorted(bundle["y_test"].unique())
    cm = confusion_matrix(bundle["y_test"], bundle["y_pred"], labels=labels)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", ax=ax2, xticklabels=labels, yticklabels=labels)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)


# ============================================================
# MODULE 7 — TRY IT YOURSELF
# ============================================================
elif page == "Try It Yourself":
    page_header("Try It Yourself", "Type a review, pick a model, get a live prediction")

    if not data_loaded:
        st.error("Dataset not found.")
        st.stop()

    user_text = st.text_area(
        "Write a movie review",
        value="This film was a total waste of time. The plot made no sense and the acting felt forced.",
        height=140,
    )
    model_choice = st.selectbox(
        "Model",
        ["SVM (LinearSVC)", "Random Forest", "KNN", "AdaBoost", "DistilBERT"],
    )

    if st.button("Predict Sentiment"):
        if not user_text.strip():
            st.warning("Type a review first.")
        elif model_choice == "DistilBERT":
            clf_pipeline = load_distilbert()
            result = clf_pipeline(user_text[:512], truncation=True)[0]
            label = "Positive" if result["label"] == "POSITIVE" else "Negative"
            st.markdown(
                f'<div class="metric-card" style="max-width:320px">'
                f'<div class="num">{label}</div>'
                f'<div class="lbl">Confidence {result["score"]:.1%}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            bundle = train_models(path, sample_size)
            vec = bundle["vectorizer"]
            model = bundle["models"][model_choice]
            cleaned = preprocess_reviews(pd.Series([user_text])).iloc[0]
            vecd = vec.transform([cleaned])
            pred = model.predict(vecd)[0]
            label = "Positive" if pred == 1 else "Negative"

            if hasattr(model, "predict_proba"):
                conf = model.predict_proba(vecd)[0].max()
                conf_str = f"{conf:.1%}"
            elif hasattr(model, "decision_function"):
                margin = model.decision_function(vecd)[0]
                conf_str = f"margin {margin:+.2f}"
            else:
                conf_str = "n/a"

            st.markdown(
                f'<div class="metric-card" style="max-width:320px">'
                f'<div class="num">{label}</div>'
                f'<div class="lbl">{model_choice} · {conf_str}</div></div>',
                unsafe_allow_html=True,
            )


# ============================================================
# MODULE 8 — CONCLUSION
# ============================================================
elif page == "Conclusion":
    page_header("Conclusion", "Summary of findings from the notebook")

    st.markdown("### Key Findings")
    st.markdown("""
    1. **TF-IDF + SVM** is the strongest classical baseline, since `LinearSVC` scales with support vectors rather than raw dimensions and fits a linear boundary cleanly in high-dimensional sparse text.
    2. **Random Forest** trails SVM by a small margin, averaging many trees over random feature subsets.
    3. **AdaBoost** sits further back: its depth-1 stumps each look at a single term out of 5,000, so it stays underfit even after hundreds of rounds.
    4. **KNN** is the weakest classical model, since distance between points stops being meaningful in a 5,000-dimensional sparse space.
    5. **DistilBERT**, run zero-shot with no training on this dataset, outperforms every classical model, because self-attention lets it read "not" as modifying "good" instead of treating every word as independent.
    6. **Emotion classification** (seven classes instead of two) is harder and more imbalanced, so macro-F1 and a majority baseline matter more than raw accuracy there.
    """)

    st.markdown("### Practical Takeaway")
    st.write(
        "For a lightweight, fast, explainable sentiment classifier, TF-IDF plus SVM is a solid choice: it trains in "
        "seconds and its coefficients name the exact words driving each prediction. For the best raw accuracy, and "
        "when latency and compute budget allow it, a pretrained transformer like DistilBERT wins with zero training."
    )
