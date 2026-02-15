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


