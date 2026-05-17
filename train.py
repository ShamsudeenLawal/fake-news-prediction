import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import extract_domain


# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/raw/fake_news.csv"
MODEL_PATH = "app/models/classifier.joblib"
SCORE_PATH = "metrics/score.json"
TEST_DATA_PATH = "data/test/test_fake_news.csv"


# -----------------------------
# DATA LOADING & PREPROCESSING
# -----------------------------
def load_and_prepare_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # rename and map labels
    df = df.rename(columns={"real": "label"})
    df["label"] = df["label"].map({0: "fake", 1: "real"})

    # remove duplicates
    df = df.drop_duplicates(keep="first")

    # feature engineering
    df["cleaned_source_domain"] = df["news_url"].apply(extract_domain)
    df["contents"] = df["cleaned_source_domain"] + " " + df["title"]

    return df


# -----------------------------
# MODEL BUILDING
# -----------------------------
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(max_iter=1000))
    ])


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_model(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)

    scores = {
        "accuracy": metrics.accuracy_score(y_test, preds),
        "precision": metrics.precision_score(y_test, preds, pos_label="real"),
        "recall": metrics.recall_score(y_test, preds, pos_label="real"),
        "f1_score": metrics.f1_score(y_test, preds, pos_label="real")
    }

    # pretty print
    print("\nModel Performance:")
    for k, v in scores.items():
        print(f"{k}: {round(v * 100, 2)}%")

    return scores


# -----------------------------
# SAVE METRICS
# -----------------------------
def save_metrics(scores: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    scores = {k: float(np.round(v, 4)) for k, v in scores.items()}

    with open(path, "w") as f:
        json.dump(scores, f, indent=4)

    print(f"\nMetrics saved to {path}")


# -----------------------------
# SAVE TEST DATA
# -----------------------------
def save_test_data(df: pd.DataFrame, X_test, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    test_df = df.loc[X_test.index, ["news_url", "title", "label"]].reset_index(drop=True)

    test_df.to_csv(path, index=False)
    test_df.to_json(path.replace(".csv", ".json"), orient="records")

    print(f"Test data saved to {path}")


# -----------------------------
# SAVE MODEL
# -----------------------------
def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    df = load_and_prepare_data(DATA_PATH)

    X = df["contents"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    scores = evaluate_model(model, X_test, y_test)

    save_metrics(scores, SCORE_PATH)
    save_model(model, MODEL_PATH)
    save_test_data(df, X_test, TEST_DATA_PATH)


if __name__ == "__main__":
    main()