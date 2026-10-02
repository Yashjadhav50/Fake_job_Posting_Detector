# 🔍 Fake Job Posting Detector

A Machine Learning and Streamlit web application that predicts whether a job posting is likely to be real or fraudulent.

## 🚀 Project Overview

Fake job postings can mislead job seekers and result in financial loss or exposure to scams.

This project uses Natural Language Processing (NLP) and Machine Learning to classify job postings based on their textual information.

The application allows users to enter job posting details and receive a prediction through an interactive Streamlit interface.

## Link: -
https://fakejobpostingdetector-by-yash.streamlit.app/

## 🧠 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Text Preprocessing
4. TF-IDF Feature Extraction
5. Train-Test Split
6. Logistic Regression
7. Model Evaluation
8. Model Serialization using Pickle
9. Streamlit Deployment

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Pickle
- Streamlit
- Jupyter Notebook

## 📊 Model

The project uses:

**TF-IDF + Logistic Regression**

Class imbalance was handled using:

```python
class_weight="balanced"
