import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, cast
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, average_precision_score
from src.preprocess import get_processed_data, get_train_test_split

st.set_page_config(page_title="Fraud Finder", layout="wide")

# --- Caching Functions ---
@st.cache_resource
def load_artifacts():
    """Loads the pre-trained model and scaler."""
    model = joblib.load("models/fraud_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    # Load column order from training
    X, _, _ = get_processed_data("data/creditcard.csv")
    return model, scaler, X.columns.tolist()

@st.cache_data
def get_test_data():
    """Loads the hold-out test set for model evaluation."""
    _, X_test, _, y_test = get_train_test_split("data/creditcard.csv")
    return X_test, y_test

# --- Load Data and Artifacts ---
model, scaler, feature_names = load_artifacts()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Live Prediction", "Model Performance"])

# ======================================================================================
# --- Live Prediction Page ---
# ======================================================================================
if page == "Live Prediction":
    st.title("💳 Live Fraud Prediction")
    st.write("Adjust the feature values below to simulate a transaction and see the model's prediction.")

    # Known important features for this dataset
    important_features = {
        'V14': -9.2, 'V10': -5.5, 'V12': -6.9, 'V17': -4.8, 'V4': 4.1, 'log_amount': 3.5
    }
    
    st.header("Transaction Features")
    col1, col2 = st.columns(2)
    with col1:
        v14 = st.slider("Feature V14", -20.0, 5.0, important_features['V14'])
        v10 = st.slider("Feature V10", -25.0, 10.0, important_features['V10'])
        v12 = st.slider("Feature V12", -20.0, 5.0, important_features['V12'])
    with col2:
        v17 = st.slider("Feature V17", -25.0, 5.0, important_features['V17'])
        v4 = st.slider("Feature V4", -5.0, 20.0, important_features['V4'])
        log_amount = st.slider("Log of Transaction Amount", 0.0, 10.0, important_features['log_amount'])

    if st.button("Predict"):
        # Create a DataFrame with default values (mean=0 after scaling)
        input_data = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Update with slider values
        input_data['V14'] = v14
        input_data['V10'] = v10
        input_data['V12'] = v12
        input_data['V17'] = v17
        input_data['V4'] = v4
        input_data['log_amount'] = log_amount
        
        # Use the pre-fitted scaler to transform the data
        # Note: The scaler expects all features, even if we only changed a few.
        scaled_features = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("🚨 Fraudulent Transaction Detected")
        else:
            st.success("✅ Transaction is Likely Legitimate")
            
        st.write(f"**Probability of Fraud:** {prediction_proba[0][1]:.2%}")
        st.write(f"**Probability of Not Fraud:** {prediction_proba[0][0]:.2%}")

# ======================================================================================
# --- Model Performance Page ---
# ======================================================================================
elif page == "Model Performance":
    st.title("📊 Model Performance Evaluation")
    st.write(
        "The following metrics are calculated on a hold-out test set (20% of the data) "
        "that the model has never seen. This provides an unbiased evaluation."
    )

    X_test, y_test = get_test_data()
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Use the optimal threshold found during training/evaluation
    optimal_threshold = 0.9965 # You can find this programmatically
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    auprc = average_precision_score(y_test, y_pred_proba)
    
    # Display key metrics
    st.header("Key Metrics")
    col1, col2, col3 = st.columns(3)
    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    with col1:
        st.metric("Fraud Precision", f"{precision:.2f}")
    with col2:
        st.metric("Fraud Recall", f"{recall:.2f}")
    with col3:
        st.metric("AUPRC", f"{auprc:.2f}")
    
    # Display plots
    st.header("Evaluation Plots")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Not Fraud', 'Fraud'], ax=ax1, colorbar=False)
    ax1.set_title("Confusion Matrix")

    PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba, ax=ax2)
    ax2.set_title("Precision-Recall Curve")
    
    plt.tight_layout()
    st.pyplot(fig)

# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Any, Dict, cast
# from sklearn.metrics import (
#     classification_report,
#     ConfusionMatrixDisplay,
#     PrecisionRecallDisplay,
#     average_precision_score
# )
# # Import the preprocessing function to get test data
# from src.preprocess import load_and_preprocess_data

# # --- Caching Functions ---

# @st.cache_resource
# def load_artifacts():
#     """Loads the pre-trained model and scaler."""
#     model = joblib.load("models/fraud_model.joblib")
#     scaler = joblib.load("models/scaler.joblib")
#     return model, scaler

# @st.cache_data
# def load_sample_data():
#     """Loads a small sample of the dataset for the UI."""
#     df = pd.read_csv("data/creditcard.csv")
#     return df.head(1000)

# @st.cache_data
# def get_test_data():
#     """
#     Loads and preprocesses the full dataset to get the hold-out test set
#     for model evaluation.
#     """
#     # We only need the test sets, so we ignore the training sets with _
#     _, X_test, _, y_test = load_and_preprocess_data("data/creditcard.csv")  # type: ignore
#     return X_test, y_test

# # --- Load Data and Artifacts ---
# model, scaler = load_artifacts()
# sample_df = load_sample_data()

# # --- Streamlit App UI ---
# st.title("💳 Credit Card Fraud Detection")
# st.write(
#     "An interactive app to predict credit card fraud using a LightGBM model. "
#     "Choose a sample transaction or view the model's overall performance."
# )

# # --- Interactive Demo Section ---
# st.header("Interactive Demo")
# st.dataframe(sample_df.head())

# selected_index = st.selectbox(
#     "Select a transaction index to test:", sample_df.index
# )
# selected_transaction = sample_df.loc[selected_index]

# st.write("You selected transaction:", selected_index)
# st.dataframe(pd.DataFrame(selected_transaction).transpose())

# if st.button("Classify Selected Transaction"):
#     # Preprocessing logic from the previous step...
#     transaction_df = pd.DataFrame(selected_transaction).transpose()
#     transaction_df["log_amount"] = np.log1p(transaction_df["Amount"])
#     seconds_in_day = 24 * 60 * 60
#     transaction_df["sin_time"] = np.sin(2 * np.pi * transaction_df["Time"] / seconds_in_day)
#     transaction_df["cos_time"] = np.cos(2 * np.pi * transaction_df["Time"] / seconds_in_day)
#     transaction_df = transaction_df.drop(["Time", "Amount"], axis=1)

#     feature_names = [f'V{i}' for i in range(1, 29)] + ['log_amount', 'sin_time', 'cos_time']
#     transaction_features = transaction_df[feature_names]

#     scaled_features = scaler.transform(transaction_features)
    
#     prediction = model.predict(scaled_features)
#     prediction_proba = model.predict_proba(scaled_features)
    
#     st.subheader("Prediction Result")
#     if prediction[0] == 1:
#         st.error("🚨 Fraudulent Transaction Detected")
#     else:
#         st.success("✅ Transaction is Likely Legitimate")
        
#     st.write(f"**Probability of Fraud:** {prediction_proba[0][1]:.2%}")
#     st.write(f"**Probability of Not Fraud:** {prediction_proba[0][0]:.2%}")

# # --- Model Performance Section ---
# st.header("Model Performance")
# with st.expander("Show model performance on the hold-out test set"):
#     st.write(
#         "The following metrics are calculated on a test set that the model "
#         "has never seen. This provides an unbiased evaluation of the model's ability "
#         "to generalize to new data."
#     )
#     X_test, y_test = get_test_data()
#     y_pred = model.predict(X_test)
#     y_pred_proba = model.predict_proba(X_test)[:, 1]
#     report = cast(Dict[str, Any], classification_report(
#         y_test, y_pred, target_names=['Not Fraud', 'Fraud'], output_dict=True
#     ))
#     auprc = average_precision_score(y_test, y_pred_proba)

#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Fraud Precision", f"{report['Fraud']['precision']:.2f}")
#     with col2:
#         st.metric("Fraud Recall", f"{report['Fraud']['recall']:.2f}")
#     with col3:
#         st.metric("AUPRC", f"{auprc:.2f}")
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#     ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Not Fraud', 'Fraud'], ax=ax1, colorbar=False)
#     ax1.set_title("Confusion Matrix")
#     PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba, ax=ax2)
#     ax2.set_title("Precision-Recall Curve")
    
#     plt.tight_layout()
#     st.pyplot(fig)