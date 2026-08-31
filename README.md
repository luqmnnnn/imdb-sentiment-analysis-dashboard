# IMDB Sentiment Analysis Dashboard

Sentiment classification on the IMDB 50K movie review dataset, comparing classical TF-IDF models against a pretrained DistilBERT running zero-shot, plus a seven-way emotion classification extension. Includes both a research notebook and an interactive Streamlit dashboard.

**Live demo:** https://imdb-sentiment-analysis-dashboard.streamlit.app/

## What's inside

- `sentiment_analysis_movie_reviews_final.ipynb` — the full analysis: preprocessing, TF-IDF feature extraction, model training, ROC-AUC, cross-validation, hyperparameter tuning, DistilBERT benchmarking, and emotion classification.
- `app.py` — a Streamlit dashboard that turns the notebook into an interactive tool: trains models live on a user-selected sample size, benchmarks against DistilBERT, and includes a page to type your own review and get a live prediction.
- `IMDB Dataset.csv` — 50,000 labeled movie reviews (positive/negative).
- `emotion_labels.csv` — cached silver emotion labels (from `j-hartmann/emotion-english-distilroberta-base`) for the emotion classification section.

## Models compared

| Model | Accuracy (full test set) |
|---|---|
| DistilBERT (zero-shot) | 89.20% (500-review subset) |
| SVM (LinearSVC) | 85.25% |
| Random Forest | 83.80% |
| AdaBoost | 79.50% |
| KNN | 71.45% |

TF-IDF + SVM is the strongest classical baseline; DistilBERT wins overall with zero training on this dataset, since self-attention lets it read negation ("not good") correctly where a bag-of-words model can't.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app expects `IMDB Dataset.csv` and `emotion_labels.csv` in the same folder. First run downloads NLTK data and the DistilBERT checkpoint automatically.

## Tech stack

Python, scikit-learn, Streamlit, Hugging Face Transformers (DistilBERT), NLTK, pandas, matplotlib/seaborn, WordCloud.
