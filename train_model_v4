import pandas as pd
import json
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier


# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv(
    r"D:\Model_train2\CT-Brain-Disease-Prediction\dataset\ct_brain_cleaned.csv"
)


# =========================================
# EXTRACT SLICE FEATURES
# =========================================

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


# =========================================
# FEATURE ENGINEERING
# =========================================

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


# =========================================
# TARGET CLEANING
# =========================================

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


# =========================================
# FEATURES
# =========================================

features = [

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
    "download_skipped_count"
]

X = df[features]

y = df["target"]


# =========================================
# LABEL ENCODING
# =========================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# =========================================
# APPLY SMOTE
# =========================================

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


print("\nAfter SMOTE Class Distribution:\n")

print(
    pd.Series(y_train_smote).value_counts()
)


# =========================================
# TUNED XGBOOST MODEL
# =========================================

model = XGBClassifier(

    n_estimators=700,

    max_depth=10,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    gamma=0.3,

    min_child_weight=3,

    objective="multi:softprob",

    num_class=len(label_encoder.classes_),

    eval_metric="mlogloss",

    random_state=42
)


# =========================================
# TRAIN MODEL
# =========================================

model.fit(
    X_train_smote,
    y_train_smote
)


# =========================================
# PREDICTIONS
# =========================================

y_pred = model.predict(X_test)


# =========================================
# EVALUATION
# =========================================

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


# =========================================
# CROSS VALIDATION
# =========================================

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y_encoded,
    cv=kfold,
    scoring="accuracy"
)

print("\nCross Validation Scores:\n")

print(cv_scores)

print("\nMean CV Accuracy:\n")

print(cv_scores.mean())


# =========================================
# SAVE MODEL
# =========================================

joblib.dump(
    model,
    "final_xgboost_model.pkl"
)

joblib.dump(
    label_encoder,
    "label_encoder.pkl"
)

print("\nFINAL MODEL SAVED SUCCESSFULLY!")