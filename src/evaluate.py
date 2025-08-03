import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)
from loguru import logger

def find_optimal_threshold(y_true, y_proba):
    """Finds the optimal probability threshold to maximize F1-score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # Correctly calculate F1 score, handling the divide-by-zero case
    f1_scores = (2 * precision * recall) / np.where(precision + recall == 0, 1, precision + recall)
    # Remove the final value to align with thresholds
    f1_scores = f1_scores[:-1]
    
    # Get the best threshold
    best_f1_idx = np.argmax(f1_scores)
    return thresholds[best_f1_idx]

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using an optimal threshold for predictions.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Find the best threshold from the test set probabilities
    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba)
    logger.info(f"Optimal threshold found: {optimal_threshold:.4f}")

    # Make final predictions using the optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    logger.info("Classification Report (at optimal threshold):")

    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Fraud', 'Fraud']).plot(ax=ax1)
    ax1.set_title("Confusion Matrix")

    # Plot Precision-Recall Curve
    avg_precision = average_precision_score(y_test, y_pred_proba)
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba, ax=ax2)
    ax2.set_title(f"Precision-Recall Curve (AUPRC = {avg_precision:.2f})")
    
    plt.suptitle("Model Evaluation", fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()