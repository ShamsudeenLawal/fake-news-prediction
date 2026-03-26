
import json
import requests
import numpy as np
import pandas as pd

# load data
df_test = pd.read_csv('data/test_fake_news.csv')

index = 0

# prepare data to send
data = {
    "news_url": df_test.at[index, "news_url"],
    "title": df_test.at[index, "title"]
    }

# get prediction
url = "http://127.0.0.1:5001/predict"
resp = requests.post(url, json=data)

response_dict = resp.json()

prediction = response_dict["prediction"]

# print predictions and true label
true_label = df_test.at[index, "label"]
print(f"Model prediction: {prediction}\nTrue Label: {true_label}")
