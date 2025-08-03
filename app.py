import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, cast
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score
)
# Import the preprocessing function to get test data
from src.preprocess import load_and_preprocess_data

# --- Caching Functions ---

@st.cache_resource
def load_artifacts():
    """Loads the pre-trained model and scaler."""
    model = joblib.load("models/fraud_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    return model, scaler

@st.cache_data
def load_sample_data():
    """Loads a small sample of the dataset for the UI."""
    df = pd.read_csv("data/creditcard.csv")
    return df.head(1000)

@st.cache_data
def get_test_data():
    """
    Loads and preprocesses the full dataset to get the hold-out test set
    for model evaluation.
    """
    # We only need the test sets, so we ignore the training sets with _
    _, X_test, _, y_test = load_and_preprocess_data("data/creditcard.csv")  # type: ignore
    return X_test, y_test

# --- Load Data and Artifacts ---
model, scaler = load_artifacts()
sample_df = load_sample_data()

# --- Streamlit App UI ---
st.title("💳 Credit Card Fraud Detection")
st.write(
    "An interactive app to predict credit card fraud using a LightGBM model. "
    "Choose a sample transaction or view the model's overall performance."
)

# --- Interactive Demo Section ---
st.header("Interactive Demo")
st.dataframe(sample_df.head())

selected_index = st.selectbox(
    "Select a transaction index to test:", sample_df.index
)
selected_transaction = sample_df.loc[selected_index]

st.write("You selected transaction:", selected_index)
st.dataframe(pd.DataFrame(selected_transaction).transpose())

if st.button("Classify Selected Transaction"):
    # Preprocessing logic from the previous step...
    transaction_df = pd.DataFrame(selected_transaction).transpose()
    transaction_df["log_amount"] = np.log1p(transaction_df["Amount"])
    seconds_in_day = 24 * 60 * 60
    transaction_df["sin_time"] = np.sin(2 * np.pi * transaction_df["Time"] / seconds_in_day)
    transaction_df["cos_time"] = np.cos(2 * np.pi * transaction_df["Time"] / seconds_in_day)
    transaction_df = transaction_df.drop(["Time", "Amount"], axis=1)

    feature_names = [f'V{i}' for i in range(1, 29)] + ['log_amount', 'sin_time', 'cos_time']
    transaction_features = transaction_df[feature_names]

    scaled_features = scaler.transform(transaction_features)
    
    prediction = model.predict(scaled_features)
    prediction_proba = model.predict_proba(scaled_features)
    
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Transaction is Likely Legitimate")
        
    st.write(f"**Probability of Fraud:** {prediction_proba[0][1]:.2%}")
    st.write(f"**Probability of Not Fraud:** {prediction_proba[0][0]:.2%}")

# --- Model Performance Section ---
st.header("Model Performance")
with st.expander("Show model performance on the hold-out test set"):
    st.write(
        "The following metrics are calculated on a test set that the model "
        "has never seen. This provides an unbiased evaluation of the model's ability "
        "to generalize to new data."
    )
    X_test, y_test = get_test_data()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    report = cast(Dict[str, Any], classification_report(
        y_test, y_pred, target_names=['Not Fraud', 'Fraud'], output_dict=True
    ))
    auprc = average_precision_score(y_test, y_pred_proba)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fraud Precision", f"{report['Fraud']['precision']:.2f}")
    with col2:
        st.metric("Fraud Recall", f"{report['Fraud']['recall']:.2f}")
    with col3:
        st.metric("AUPRC", f"{auprc:.2f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Not Fraud', 'Fraud'], ax=ax1, colorbar=False)
    ax1.set_title("Confusion Matrix")
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba, ax=ax2)
    ax2.set_title("Precision-Recall Curve")
    
    plt.tight_layout()
    st.pyplot(fig)