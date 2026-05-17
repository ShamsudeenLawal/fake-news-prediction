from pathlib import Path
import joblib
import pandas as pd
import logging

from fastapi import FastAPI
from pydantic import BaseModel, Field

from utils import extract_domain

# -----------------------------
# LOGGING SETUP
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI(
    title="Fake News Classifier API",
    description="Predict whether a news article is fake or real using ML",
    version="1.0.0"
)


# -----------------------------
# MODEL LOADING
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "classifier.joblib"

model = joblib.load(MODEL_PATH)
logger.info("Model loaded successfully")


# -----------------------------
# SCHEMAS
# -----------------------------
class News(BaseModel):
    news_url: str | None = Field(default="")
    title: str


class BatchNews(BaseModel):
    batch_news: list[News]


class PredictionResponse(BaseModel):
    prediction: str


class BatchPredictionResponse(BaseModel):
    predictions: list[str]


# -----------------------------
# PREPROCESSING FUNCTION
# -----------------------------
def preprocess_text(news_url: str | None, title: str) -> str:
    domain = extract_domain(news_url or "")
    return f"{domain} {title}"


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Fake News Classifier API is running 🚀"
    }


# -----------------------------
# SINGLE PREDICTION
# -----------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(news: News):

    content = preprocess_text(news.news_url, news.title)
    prediction = model.predict([content])[0]

    logger.info(f"Single prediction: {prediction}")

    return {"prediction": prediction}


# -----------------------------
# BATCH PREDICTION
# -----------------------------
@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(news: BatchNews):

    df = pd.DataFrame([item.model_dump() for item in news.batch_news])

    df["content"] = df.apply(
        lambda row: preprocess_text(row["news_url"], row["title"]),
        axis=1
    )

    predictions = model.predict(df["content"].tolist()).tolist()

    logger.info(f"Batch prediction count: {len(predictions)}")

    return {"predictions": predictions}


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        port=5001,
        reload=True
    )
