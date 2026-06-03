import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("D:\Model_train2\CT-Brain-Disease-Prediction\dataset\ct_brain_cleaned.csv")


# =========================
# EXTRACT FEATURES
# =========================

def extract_slice_info(x):

    try:
        data = json.loads(x)

        all_slices = []

        for key in data:
            slices = data[key]

            for s in slices:
                all_slices.append(int(s))

        if len(all_slices) == 0:
            return pd.Series([0, 0, 0])

        return pd.Series([
            len(all_slices),
            min(all_slices),
            max(all_slices)
        ])

    except:
        return pd.Series([0, 0, 0])


df[[
    "slice_count",
    "first_slice",
    "last_slice"
]] = df["selected_series_slices_json"].apply(extract_slice_info)


# =========================
# ADD EXTRA FEATURES
# =========================

df["slice_span"] = (
    df["last_slice"] - df["first_slice"]
)

df["abnormal_ratio"] = (
    df["slice_count"] /
    (df["downloaded_slices"] + 1)
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
    "downloaded_slices",
    "download_failed_count",
    "download_skipped_count"
]

X = df[features]

y = df["target"]


# =========================
# ENCODE TARGET
# =========================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# =========================
# XGBOOST MODEL
# =========================

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    random_state=42
)


# =========================
# TRAIN MODEL
# =========================

model.fit(X_train, y_train)


# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test)


# =========================
# EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:\n")
print(accuracy)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(y_test, y_pred)
)


# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "xgboost_model.pkl")

joblib.dump(label_encoder, "label_encoder.pkl")

print("\nXGBoost Model Saved Successfully!")