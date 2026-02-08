import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Page config
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("Heart Disease Prediction using Machine Learning")

# Load scaler
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))

# Load models
models = {
    "Logistic Regression": joblib.load(os.path.join(BASE_DIR, "model", "logistic_regression.pkl")),
    "Decision Tree": joblib.load(os.path.join(BASE_DIR, "model", "decision_tree.pkl")),
    "KNN": joblib.load(os.path.join(BASE_DIR, "model", "knn.pkl")),
    "Naive Bayes": joblib.load(os.path.join(BASE_DIR, "model", "naive_bayes.pkl")),
    "Random Forest": joblib.load(os.path.join(BASE_DIR, "model", "random_forest.pkl")),
    "XGBoost": joblib.load(os.path.join(BASE_DIR, "model", "xgboost.pkl")),
}

# ================= SIDEBAR =================

# Download sample CSV
st.sidebar.markdown("### Download Sample Test Data")
with open(os.path.join(BASE_DIR, "heart_disease.csv"), "rb") as file:
    st.sidebar.download_button(
        label="Download Test CSV",
        data=file,
        file_name="heart_disease.csv",
        mime="text/csv"
    )

# Upload CSV
st.sidebar.markdown("### Upload Test Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (target column must be last)",
    type=["csv"]
)

# Model selection
model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))

# ================= MAIN APP =================

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Rename columns to match training
    df = df.rename(columns={
        "cp": "chest_pain",
        "chol": "cholesterol",
        "restecg": "ecg",
        "exang": "exercise_angina",
        "fbs": "fast_sugar",
        "trestbps": "rest_bp",
        "thalach": "max_hr",
        "ca": "vessels"
    })

    # Split features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Scale features
    X_scaled = scaler.transform(X)

    # Model prediction
    model = models[model_choice]
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Metrics
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("AUC", f"{auc:.4f}")
    col3.metric("Precision", f"{prec:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{rec:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )
    st.dataframe(cm_df)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.success(f"Evaluation completed using {model_choice}")

else:
    st.info("Please download the sample CSV or upload your own test dataset to begin.")
