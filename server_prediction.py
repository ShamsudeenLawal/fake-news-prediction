import requests
import pandas as pd


# -----------------------------
# CONFIG
# -----------------------------
FILEPATH = "data/test/test_fake_news.csv"
API_URL = "http://127.0.0.1:5001/predict"
INDEX = 0


# -----------------------------
# LOAD TEST DATA
# -----------------------------
df_test = pd.read_csv(FILEPATH)


# -----------------------------
# PREPARE REQUEST DATA
# -----------------------------
data = {
    "news_url": df_test.at[INDEX, "news_url"],
    "title": df_test.at[INDEX, "title"]
}


# -----------------------------
# SEND REQUEST
# -----------------------------
try:
    response = requests.post(
        API_URL,
        json=data,
        timeout=30
    )

    response.raise_for_status()

except requests.exceptions.RequestException as e:
    print(f"❌ Request failed: {e}")
    raise SystemExit()


# -----------------------------
# FETCH PREDICTION
# -----------------------------
result = response.json()

if "prediction" not in result:
    print("❌ No prediction returned from API")
    raise SystemExit()

prediction = result["prediction"]


# -----------------------------
# FETCH TRUE LABEL
# -----------------------------
true_label = df_test.at[INDEX, "label"]


# -----------------------------
# DISPLAY RESULTS
# -----------------------------
print("\n📰 News Title:")
print(df_test.at[INDEX, "title"])

print("\n🔮 Model Prediction:")
print(prediction)

print("\n✅ True Label:")
print(true_label)

print("\n🎯 Correct Prediction:")
print(prediction == true_label)