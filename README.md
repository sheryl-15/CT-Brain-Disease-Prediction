# CT Brain Disease Classification

## Objective

Predict CT brain disease categories using metadata and abnormal slice information.

## Dataset

3555 CT Brain Records

Classes:

- NORMAL
- INFARCT
- ISCHEMIC
- HEMORRHAGE
- FRACTURE
- MASS
- HYDROCEPHALUS
- SWELLING
- OTHER

## Features

- slice_count
- first_slice
- last_slice
- slice_span
- abnormal_ratio
- downloaded_slices

## Model

Random Forest Classifier

## Results

Accuracy: 74.26%

## Future Work

- TF-IDF Text Features
- XGBoost
- Disease Prediction Web Application