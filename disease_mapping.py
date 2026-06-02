import pandas as pd

df = pd.read_csv("dataset/ct_brain_labels.csv")

def map_disease(text):
    text = str(text).upper()

    if "INFARCT" in text:
        return "INFARCT"

    elif "HEMORRHAGE" in text or "SDH" in text or "SAH" in text or "EDH" in text or "BLEED" in text:
        return "HEMORRHAGE"

    elif "FRACTURE" in text:
        return "FRACTURE"

    elif "ISCHEM" in text:
        return "ISCHEMIC"

    elif "ENCEPHALOMALACIA" in text:
        return "ENCEPHALOMALACIA"

    elif "HYDROCEPHALUS" in text:
        return "HYDROCEPHALUS"

    elif "MASS" in text:
        return "MASS"

    elif "GLIOSIS" in text:
        return "GLIOSIS"

    elif "CYST" in text:
        return "CYST"

    elif "ATROPHY" in text:
        return "ATROPHY"

    elif "CALCIFICATION" in text or "CALCIFIED" in text:
        return "CALCIFICATION"

    elif "SWELLING" in text or "HEMATOMA" in text:
        return "SWELLING"

    elif "NORMAL" in text:
        return "NORMAL"

    else:
        return "OTHER"

df["disease_clean"] = df["disease"].apply(map_disease)

print(df["disease_clean"].value_counts())
df["disease_clean"] = df["disease"].apply(map_disease)

df.to_csv(
    "dataset/ct_brain_cleaned.csv",
    index=False
)

print("Cleaned dataset saved")

print("\nUnique Cleaned Diseases:")
print(sorted(df["disease_clean"].unique()))

print("\nNumber of Cleaned Diseases:")
print(df["disease_clean"].nunique())

print("\nDisease Distribution:")
print(df["disease_clean"].value_counts())