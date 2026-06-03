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

from catboost import CatBoostClassifier


# ==================================
# LOAD DATASET
# ==================================

df = pd.read_csv(
    r"D:\Model_train2\CT-Brain-Disease-Prediction\dataset\ct_brain_cleaned.csv"
)


# ==================================
# EXTRACT SLICE FEATURES
# ==================================

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


# ==================================
# FEATURE ENGINEERING
# ==================================

df["slice_span"] = (
    df["last_slice"] - df["first_slice"]
)

df["abnormal_ratio"] = (
    df["slice_count"] /
    (df["downloaded_slices"] + 1)
)

df["slice_density"] = (
    df["slice_count"] /
    (df["slice_span"] + 1)
)

df["center_slice"] = (
    df["first_slice"] +
    df["last_slice"]
) / 2

df["is_abnormal"] = (
    df["slice_count"] > 0
).astype(int)


# ==================================
# TARGET CLEANING
# ==================================

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


# ==================================
# SELECT FEATURES
# ==================================

features = [

    # Numerical
    "slice_count",
    "first_slice",
    "last_slice",
    "slice_span",
    "abnormal_ratio",
    "slice_density",
    "center_slice",
    "is_abnormal",
    "downloaded_slices",
    "download_failed_count",
    "download_skipped_count",

    # Categorical/Text
    "study_description",
    "selected_series_description",
    "contrast_type"
]

X = df[features]

y = df["target"]


# ==================================
# HANDLE MISSING VALUES
# ==================================

X["selected_series_description"] = (
    X["selected_series_description"]
    .fillna("UNKNOWN")
)

X["study_description"] = (
    X["study_description"]
    .fillna("UNKNOWN")
)

X["contrast_type"] = (
    X["contrast_type"]
    .fillna("UNKNOWN")
)


# ==================================
# LABEL ENCODING TARGET
# ==================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


# ==================================
# TRAIN TEST SPLIT
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# ==================================
# CATEGORICAL FEATURE INDEXES
# ==================================

categorical_features = [
    "study_description",
    "selected_series_description",
    "contrast_type"
]

cat_features_index = [
    X.columns.get_loc(col)
    for col in categorical_features
]


# ==================================
# CATBOOST MODEL
# ==================================

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    verbose=100,
    random_seed=42
)


# ==================================
# TRAIN MODEL
# ==================================

model.fit(
    X_train,
    y_train,
    cat_features=cat_features_index
)


# ==================================
# PREDICTIONS
# ==================================

y_pred = model.predict(X_test)

y_pred = y_pred.flatten()


# ==================================
# EVALUATION
# ==================================

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


# ==================================
# SAVE MODEL
# ==================================

joblib.dump(model, "catboost_model.pkl")

joblib.dump(label_encoder, "label_encoder.pkl")

print("\nCatBoost Model Saved Successfully!")