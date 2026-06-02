import pandas as pd
import json

df = pd.read_csv("dataset/ct_brain_cleaned.csv")

# Feature 1
def extract_slice_count(x):
    try:
        data = json.loads(str(x))
        count = 0
        for value in data.values():
            count += len(value)
        return count
    except:
        return 0

# Feature 2
def first_slice(x):
    try:
        data = json.loads(str(x))
        for value in data.values():
            return int(value[0])
        return 0
    except:
        return 0

# Feature 3
def last_slice(x):
    try:
        data = json.loads(str(x))
        for value in data.values():
            return int(value[-1])
        return 0
    except:
        return 0

df["slice_count"] = df["selected_series_slices_json"].apply(extract_slice_count)
df["first_slice"] = df["selected_series_slices_json"].apply(first_slice)
df["last_slice"] = df["selected_series_slices_json"].apply(last_slice)

# ADD THESE LINES HERE 👇

print(df[[
    "slice_count",
    "first_slice",
    "last_slice"
]].describe())

print("\nDisease-wise Average Slice Count:\n")

disease_stats = df.groupby("disease_clean")["slice_count"].mean()

print(disease_stats.sort_values(ascending=False))