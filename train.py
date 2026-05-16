import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import extract_domain


# load dataset
df = pd.read_csv("data/fake_news.csv")

# rename real column to target column
df = df.rename(columns={"real": "label"})

# map target values to strings (scikit-learn supports string labels)
labels = ["fake", "real"]
label_map = {
    0: "fake",
    1: "real"
}

df["label"] = df["label"].map(label_map)

# drop duplicates
df = df.drop_duplicates(keep="first")

# extract source domain from news_url and take care of missing url


df["cleaned_source_domain"] = df["news_url"].apply(extract_domain)

# create training features
df["contents"] = df["cleaned_source_domain"] + " " + df["title"]

# fetch features and target variables
X = df["contents"]
y = df["label"]

# split dataset
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# create pipeline
pipe = Pipeline([
    ("transformer", TfidfVectorizer()),
    ("classifier", LogisticRegression())
    ])

# train model
pipe.fit(Xtrain, ytrain)

# evaluate model
test_pred = pipe.predict(Xtest)
accuracy = metrics.accuracy_score(ytest, test_pred)
recall = metrics.recall_score(ytest, test_pred, pos_label="real")
precision = metrics.precision_score(ytest, test_pred, pos_label="real")
f1 = metrics.f1_score(ytest, test_pred, pos_label="real")

print(f"accuracy: {100 * np.round(accuracy, decimals=3)}%")
print(f"precision: {100 * np.round(precision, decimals=3)}%")
print(f"recall: {100 * np.round(recall, decimals=3)}%")
print(f"f1: {100 * np.round(f1, decimals=3)}%")

# ## classification report
# report = metrics.classification_report(ytest, test_pred)
# print(report)

# ## confusion matrix
# cm = metrics.confusion_matrix(ytest, test_pred)
# cm_plot = metrics.ConfusionMatrixDisplay(cm, display_labels=list(label_map.values()))
# cm_plot.plot()
# plt.show()

# serialize model
model_path = "app/models/classifier.joblib"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(pipe, model_path)


# # Save test data
# fetch test 
df_test = df.loc[Xtest.index][["news_url", "title", "label"]]

# reset data index
df_test = df_test.reset_index(drop=True)

# save to csv and json
os.makedirs("data", exist_ok=True)
df_test.to_csv("data/test_fake_news.csv", index=False)
df_test.to_json("data/test_fake_news.json", index=False)
