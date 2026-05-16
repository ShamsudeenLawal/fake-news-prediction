# load packages
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from utils import extract_domain

# instantiate app
app = FastAPI()

# load model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "classifier.joblib"
model = joblib.load(MODEL_PATH)

# define Body Model
class News(BaseModel):
    news_url: str | None = None
    title: str

class BatchNews(BaseModel):
    batch_news: list[News]

# define endpoints
@app.get("/")
def home():
    return {"message": "Your model is live, let's get the news verified."}

@app.post("/predict")
def predict(news: News):
    news_url = news.news_url or ""
    title = news.title

    domain_name = extract_domain(news_url)
    content = [domain_name + " " + title]

    pred = model.predict(content)[0]

    return {"prediction": pred}

@app.post("/predict-batch")
def predict_batch(news: BatchNews):
    data = [news_item.model_dump() for news_item in news.batch_news]
    df = pd.DataFrame(data=data)

    df["news_url"] = df["news_url"].fillna("").map(extract_domain)
    df["content"] = df["news_url"] + " " + df["title"]

    content = df["content"].tolist()
    print(content[0])
    predictions = model.predict(content).tolist()

    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=5001, log_level="info")