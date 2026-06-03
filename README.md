# CT Brain Disease Classification using Metadata and Machine Learning

## Project Overview

This project focuses on predicting CT Brain disease categories using CT scan metadata and abnormal slice information instead of actual CT scan images.

The system analyzes metadata extracted from CT brain studies and predicts possible disease categories using Machine Learning models.

This project was developed as part of an AI/ML internship project in medical imaging analysis.

---

# Objective

Build a Machine Learning model capable of classifying CT Brain studies into clinically meaningful disease categories using metadata features.

---

# Dataset Information

* Total Records: **3555 CT Brain Studies**
* Dataset Type: **Metadata-based**
* Data Source: CT Brain Study Metadata

---

# Final Disease Classes

The final optimized disease categories are:

* NORMAL
* STROKE
* HEMORRHAGE
* FRACTURE
* OTHER

---

# Important Note

This project currently uses:

* CT scan metadata
* abnormal slice information
* study descriptions

and **does not yet use actual CT scan images**.

---

# Features Used

The following engineered features were used for training:

* slice_count
* first_slice
* last_slice
* slice_range
* downloaded_slices
* download_failed_count
* download_skipped_count

---

# Disease Cleaning and Optimization

The original dataset contained:

* noisy disease labels
* spelling inconsistencies
* overlapping disease categories

The dataset was cleaned and optimized by:

* standardizing disease names
* merging overlapping diseases
* reducing inter-class ambiguity

Example:

* INFARCT
* ISCHEMIC
* ISCHEMIA

were merged into:

* STROKE

This significantly improved model performance.

---

# Machine Learning Models Tried

The following models were trained and evaluated:

1. Random Forest Classifier
2. XGBoost Classifier
3. TF-IDF + XGBoost
4. CatBoost Classifier
5. SMOTE + XGBoost

---

# Best Performing Model

## XGBoost Classifier

Final Accuracy Achieved:

# 83.8%

---

# Model Performance Improvements

Initial Accuracy:

* Random Forest → 74%

Final Accuracy:

* Optimized XGBoost → 83.8%

Performance improved mainly through:

* disease label cleaning
* feature engineering
* class optimization
* metadata preprocessing

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* CatBoost
* Joblib
* Git & GitHub

---

# Project Workflow

CT Metadata
→ Data Cleaning
→ Feature Engineering
→ Disease Mapping
→ Model Training
→ Disease Prediction

---

# Future Work

Future improvements for this project include:

* Streamlit Web Application
* Real CT Image Analysis
* Deep Learning Models (CNN, ResNet50)
* TensorFlow Integration
* Automated Medical Report Generation
* Multi-Disease Prediction Dashboard

---

# Repository Structure

```plaintext
CT-Brain-Disease-Prediction/
│
├── dataset/
├── train_model_v1.py
├── train_model_v2.py
├── train_model_v3.py
├── train_model_v4.py
├── final_model.pkl
├── label_encoder.pkl
├── README.md
```

---

# Author

CT Brain Disease Classification Internship Project
