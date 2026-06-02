import pandas as pd

df = pd.read_csv("dataset/ct_brain_cleaned.csv")

print("Rows and Columns:")
print(df.shape)

print("\nColumns:")
print(df.columns)



print("\nUnique Cleaned Diseases:")
print(sorted(df["disease_clean"].unique()))

print("\nNumber of Cleaned Diseases:")
print(df["disease_clean"].nunique())

print("\nDisease Distribution:")
print(df["disease_clean"].value_counts())

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