import pandas as pd

df = pd.read_csv("dataset/ct_brain_cleaned.csv")

print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

import pandas as pd

df = pd.read_csv("dataset/ct_brain_cleaned.csv")

print("\nSelected Series Description:")
print(df["selected_series_description"].value_counts().head(20))

print("\nStudy Description:")
print(df["study_description"].value_counts().head(20))

print("\nContrast Type:")
print(df["contrast_type"].value_counts())