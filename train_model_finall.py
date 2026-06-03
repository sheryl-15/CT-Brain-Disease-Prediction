import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(
    r"D:\Model_train2\CT-Brain-Disease-Prediction\dataset\ct_brain_cleaned.csv"
)

# =========================
# CREATE SLICE FEATURES
# =========================

def extract_slice_info(slice_json):

    try:
        data = json.loads(slice_json)

        all_slices = []

        for key in data:
            all_slices.extend(
                [int(x) for x in data[key]]
            )

        if len(all_slices) == 0:
            return 0, 0, 0

        return (
            len(all_slices),
            min(all_slices),
            max(all_slices)
        )

    except:
        return 0, 0, 0

df[
    ["slice_count", "first_slice", "last_slice"]
] = df[
    "selected_series_slices_json"
].apply(
    lambda x: pd.Series(
        extract_slice_info(x)
    )
)

# =========================
# NEW FEATURE
# =========================

df["slice_range"] = (
    df["last_slice"] - df["first_slice"]
)

# =========================
# CREATE FINAL TARGET
# =========================

def create_target(disease):

    disease = str(disease).upper()

    # NORMAL
    if "NORMAL" in disease:
        return "NORMAL"

    # STROKE
    elif (
        "INFARCT" in disease
        or "ISCHEMIC" in disease
        or "ISCHEMIA" in disease
    ):
        return "STROKE"

    # HEMORRHAGE
    elif (
        "HEMORRHAGE" in disease
        or "HAEMORRHAGE" in disease
        or "BLEED" in disease
        or "SDH" in disease
        or "EDH" in disease
        or "SAH" in disease
    ):
        return "HEMORRHAGE"

    # FRACTURE
    elif "FRACTURE" in disease:
        return "FRACTURE"

    # EVERYTHING ELSE
    else:
        return "OTHER"

# APPLY TARGET
df["target"] = df["disease"].apply(create_target)

# =========================
# CHECK CLASS DISTRIBUTION
# =========================

print("\nFinal Disease Distribution:\n")
print(df["target"].value_counts())

# =========================
# FEATURES
# =========================

X = df[
    [
        "slice_count",
        "first_slice",
        "last_slice",
        "slice_range",
        "downloaded_slices",
        "download_failed_count",
        "download_skipped_count"
    ]
]

y = df["target"]

# =========================
# LABEL ENCODING
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
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    eval_metric='mlogloss'
)

# TRAIN
model.fit(X_train, y_train)

# PREDICT
y_pred = model.predict(X_test)

# =========================
# RESULTS
# =========================

accuracy = accuracy_score(
    y_test,
    y_pred
)

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
    confusion_matrix(
        y_test,
        y_pred
    )
)

# =========================
# SAVE MODEL
# =========================

import joblib

joblib.dump(
    model,
    "xgboost_v4.pkl"
)

joblib.dump(
    label_encoder,
    "label_encoder_v4.pkl"
)

print("\nMODEL SAVED SUCCESSFULLY!")