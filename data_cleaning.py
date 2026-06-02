import pandas as pd

df = pd.read_csv("dataset/ct_brain_cleaned.csv")

print(df["selected_series_slices_json"].iloc[0])

print("\n-----------------\n")

print(df["selected_series_slices_json"].iloc[1])

print("\n-----------------\n")

print(df["selected_series_slices_json"].iloc[2])