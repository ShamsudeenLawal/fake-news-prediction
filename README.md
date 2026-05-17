# 📰 Fake News Prediction System

An end-to-end machine learning system for detecting fake news using Natural Language Processing (NLP) and Logistic Regression. The project includes data preprocessing, feature engineering, model training, evaluation, and deployment via FastAPI with batch and real-time prediction support.

---

## 📌 Project Overview

This system classifies news articles as **fake** or **real** using textual and metadata signals such as:

- News title
- Source domain extracted from URL

It is built as a **production-ready ML service** with REST APIs for real-time and batch inference.

---

## 🧠 Problem Statement

Given a news article (title + URL), predict whether the news is:

- 🟥 Fake
- 🟩 Real

---

## 📊 Dataset

- Source: Kaggle Fake News Dataset
- Format: CSV containing news metadata
- Key columns:
  - `title`
  - `news_url`
  - `label` (0 = fake, 1 = real)

📥 Dataset Link:  
https://storage.googleapis.com/kaggle-data-sets/2623949/4484183/bundle/archive.zip

---

## 🧪 Machine Learning Pipeline

### 🔹 Data Preprocessing

- Loaded dataset using `pandas`
- Renamed target column (`real → label`)
- Mapped labels:
  - `0 → fake`
  - `1 → real`
- Removed duplicate entries
- Extracted domain from `news_url`
- Combined features:

  `content = domain + " " + title`

---

### 🔹 Feature Engineering

- TF-IDF Vectorization (`TfidfVectorizer`)
- Input feature: combined text (domain + title)

---

### 🔹 Model Training

- Algorithm: Logistic Regression
- Pipeline:

TF-IDF → LogisticRegression

- Train/Test split: 80/20 (stratified)

---

### 🔹 Evaluation Metrics

- Accuracy
- Precision (real class)
- Recall (real class)
- F1-score (real class)

---

### 📊 Model Performance (Example Output)

- Accuracy: 98.2%
- Strong performance on "real" class detection

---

## 🚀 API Deployment (FastAPI)

The model is served using FastAPI with both single and batch prediction support.

---

### ▶️ Start Server

```bash
python app/main.py
```

Server runs at:

`http://localhost:5001`

#### 📡 API Endpoints

- 🏠 Root Endpoint
  GET `/`

Response:

```json
{
  "message": "Your model is live, let's get the news verified."
}
```

- 🔍 Single Prediction
  POST `/predict`

Request:

```json
{
  "news_url": "https://example.com/news/article",
  "title": "Government announces new policy on education"
}
```

Response:

```json
{
  "prediction": "real"
}
```

- 📦 Batch Prediction
  POST `/predict-batch`

Request:

```json
{
  "batch_news": [
    {
      "news_url": "https://example.com/news/1",
      "title": "Breaking news headline one"
    },
    {
      "news_url": "https://example.com/news/2",
      "title": "Breaking news headline two"
    }
  ]
}
```

Response:

```json
{
  "predictions": ["fake", "real"]
}
```

🧪 Model Testing

You can test the API using request scripts:

- no batch prediction

```bash
python server_prediction.py
```

- batch prediction

```bash
python batch_server_prediction.py
```

## 🧠 Tech Stack

- Python
- Scikit-learn
- Pandas / NumPy
- FastAPI
- NLP (TF-IDF)

## 🔧 Key Design Decisions

- Combined URL domain + title improves classification signal
- TF-IDF used for strong baseline NLP performance
- Logistic Regression chosen for:
  - speed
  - interpretability
  - strong baseline accuracy
- Batch inference added for scalability
