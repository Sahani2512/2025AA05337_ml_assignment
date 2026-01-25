import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("Heart Disease Prediction using Machine Learning")

# Load scaler and models
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))

models = {
    "Logistic Regression": joblib.load(os.path.join(BASE_DIR, "model", "logistic_regression.pkl")),
    "Decision Tree": joblib.load(os.path.join(BASE_DIR, "model", "decision_tree.pkl")),
    "KNN": joblib.load(os.path.join(BASE_DIR, "model", "knn.pkl")),
    "Naive Bayes": joblib.load(os.path.join(BASE_DIR, "model", "naive_bayes.pkl")),
    "Random Forest": joblib.load(os.path.join(BASE_DIR, "model", "random_forest.pkl")),
    "XGBoost": joblib.load(os.path.join(BASE_DIR, "model", "xgboost.pkl")),
}

st.sidebar.header("Input Patient Data")

age = st.sidebar.number_input("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex (0=Female,1=Male)", [0, 1])
cp = st.sidebar.number_input("Chest Pain Type", 0, 3, 1)
trestbps = st.sidebar.number_input("Resting BP", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar >120 (0/1)", [0, 1])
restecg = st.sidebar.number_input("Rest ECG", 0, 2, 1)
thalach = st.sidebar.number_input("Max Heart Rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina (0/1)", [0, 1])
oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.sidebar.number_input("Slope", 0, 2, 1)
ca = st.sidebar.number_input("CA", 0, 4, 0)
thal = st.sidebar.number_input("Thal", 0, 3, 2)

model_choice = st.selectbox("Choose Model", list(models.keys()))

input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    model = models[model_choice]

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 0:
        st.error(f"Heart Disease Detected (Probability: {1-prob:.2f})")
    else:
        st.success(f"No Heart Disease Detected (Probability: {prob:.2f})")
