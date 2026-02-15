import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Page Title
st.title("Breast Cancer Detection App")

st.write("Upload test dataset (CSV format).")


# Model Selection
model_option = st.selectbox(
    "Select Model",
    ("Logistic Regression", "Random Forest", "XGBoost", "Decision Tree", "Naive Bayes", "K-Nearest Neighbor")
)

# Load Model Function
def load_model(model_name):
    if model_name == "Logistic Regression":
        model_path = os.path.join(BASE_DIR, "model", "logistic_regression.pkl")
        return joblib.load(model_path)
    elif model_name == "Random Forest":
        model_path = os.path.join(BASE_DIR, "model", "random_forest.pkl")
        return joblib.load(model_path)
    elif model_name == "XGBoost":
        model_path = os.path.join(BASE_DIR, "model", "xgboost.pkl")
        return joblib.load(model_path)
    elif model_name == "K-Nearest Neighbor":
        model_path = os.path.join(BASE_DIR, "model", "k-nearest_neighbor.pkl")
        return joblib.load(model_path)
    elif model_name == "Decision Tree":
        model_path = os.path.join(BASE_DIR, "model", "decision_tree.pkl")
        return joblib.load(model_path)
    elif model_name == "Naive Bayes":
        model_path = os.path.join(BASE_DIR, "model", "naive_bayes_(gaussian).pkl")
        return joblib.load(model_path)

# File Upload
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(data.head())


    # Prepare Data
    if "Diagnosis" not in data.columns:
        st.error("CSV must contain 'Diagnosis' column.")
    else:
        X_test = data.drop("Diagnosis", axis=1)
        y_test = data["Diagnosis"]
        print("NaNs in y_test:", y_test.isna().sum())
        print("Total y_test length:", len(y_test))
        model = load_model(model_option)

        # Predictions
        y_pred = model.predict(X_test)
        print(f"{y_pred}")

        # If model supports probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = "Not Available"


        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader("Evaluation Metrics")
        st.write(f"Model Name: {model_option}")
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1 Score: {f1:.4f}")
        st.write(f"AUC: {auc}")
        st.write(f"MCC: {mcc:.4f}")


        # Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        ax.matshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha='center', va='center')

        st.pyplot(fig)

