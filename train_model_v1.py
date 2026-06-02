import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("dataset/ct_brain_cleaned.csv")

# =========================
# FEATURE ENGINEERING
# =========================

def extract_slice_count(x):
    try:
        data = json.loads(str(x))
        count = 0

        for value in data.values():
            count += len(value)

        return count

    except:
        return 0


def first_slice(x):
    try:
        data = json.loads(str(x))

        for value in data.values():
            return int(value[0])

        return 0

    except:
        return 0


def last_slice(x):
    try:
        data = json.loads(str(x))

        for value in data.values():
            return int(value[-1])

        return 0

    except:
        return 0


df["slice_count"] = df["selected_series_slices_json"].apply(
    extract_slice_count
)

df["first_slice"] = df["selected_series_slices_json"].apply(
    first_slice
)

df["last_slice"] = df["selected_series_slices_json"].apply(
    last_slice
)

df["slice_span"] = (
    df["last_slice"] - df["first_slice"]
)

df["abnormal_ratio"] = (
    df["slice_count"] /
    df["downloaded_slices"]
)

# =========================
# TARGET CLASSES
# =========================

small_classes = [
    "ATROPHY",
    "CALCIFICATION",
    "GLIOSIS",
    "ENCEPHALOMALACIA",
    "CYST"
]

df["target"] = df["disease_clean"].replace(
    small_classes,
    "OTHER"
)

# =========================
# FEATURES
# =========================

features = [
    "slice_count",
    "first_slice",
    "last_slice",
    "slice_span",
    "abnormal_ratio",
    "downloaded_slices"
]

X = df[features]

y = df["target"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# RANDOM FOREST
# =========================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# PREDICTION
# =========================

pred = model.predict(X_test)

# =========================
# RESULTS
# =========================

accuracy = accuracy_score(
    y_test,
    pred
)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        pred
    )
)

print("\nConfusion Matrix:\n")
print(
    confusion_matrix(
        y_test,
        pred
    )
)

# =========================
# SAVE MODEL
# =========================

joblib.dump(
    model,
    "ct_brain_model.pkl"
)

print(
    "\nModel Saved Successfully!"
)