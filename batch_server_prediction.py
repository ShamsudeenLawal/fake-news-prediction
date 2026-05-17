import os
from pathlib import Path
import requests
import pandas as pd
from sklearn.metrics import accuracy_score

from utils import load_test_data


# -----------------------------
# CONFIG
# -----------------------------
FILEPATH = "data/test/test_fake_news.json"
API_URL = "http://127.0.0.1:5001/predict-batch"
OUTPUT_PATH = "data/server/server_batch_predictions.csv"


# -----------------------------
# LOAD TEST DATA
# -----------------------------
payload, test_df = load_test_data(FILEPATH)

# format payload for API
payload = {"batch_news": payload}


# -----------------------------
# SEND REQUEST
# -----------------------------
try:
    response = requests.post(
        API_URL,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

except requests.exceptions.RequestException as e:
    print(f"❌ API request failed: {e}")
    raise SystemExit()


print(f"✅ Status Code: {response.status_code}")


# -----------------------------
# FETCH PREDICTIONS
# -----------------------------
result = response.json()

if "predictions" not in result:
    print("❌ No predictions returned from API")
    raise SystemExit()

predictions = result["predictions"]


# -----------------------------
# COMBINE RESULTS
# -----------------------------
predicted_df = test_df.copy()
predicted_df["predictions"] = predictions


# -----------------------------
# SAVE PREDICTIONS
# -----------------------------
Path(os.path.dirname(OUTPUT_PATH)).mkdir(
    parents=True,
    exist_ok=True
)

predicted_df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Predictions saved to:\n{OUTPUT_PATH}")


# -----------------------------
# EVALUATE RESULTS
# -----------------------------
score = accuracy_score(
    predicted_df["label"],
    predicted_df["predictions"]
)

print(f"\n🎯 Accuracy Score: {score:.4f}")


# -----------------------------
# PREVIEW RESULTS
# -----------------------------
print("\n🔍 Sample Predictions:")
print(predicted_df.head(20))