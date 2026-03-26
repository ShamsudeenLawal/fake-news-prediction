
import json
import requests
import numpy as np
import pandas as pd
from utils import load_test_data

# load data
filepath = "data/test_fake_news.json"
payload, test_df = load_test_data(filepath)

payload = {"batch_news": payload}

# get prediction
url = "http://127.0.0.1:5001/predict-batch"
resp = requests.post(url, json=payload)
print(resp.status_code)

# # fetch and match with data to compare
predictions = resp.json()["predictions"]

predictions_df = pd.DataFrame(data=predictions, columns=["predictions"])
predicted_df = pd.concat([test_df, predictions_df], axis=1)

# save predictions to file
predicted_df.to_csv("data/server_batch_predictions.csv", index=False)

print(predicted_df.head(20))

from sklearn.metrics import accuracy_score

score = accuracy_score(predicted_df["label"], predicted_df["predictions"])
print(score)