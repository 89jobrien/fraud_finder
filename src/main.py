from sklearn.model_selection import train_test_split
from src.preprocess import load_and_preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from loguru import logger

def run_pipeline():
    """
    Runs the full data processing, model training, and evaluation pipeline.
    """
    data_path = "data/creditcard.csv"

    logger.info("Step 1: Loading and preprocessing data...")
    X, y, _ = load_and_preprocess_data(data_path)

    logger.info("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Data successfully split.")

    logger.info("\nStep 2: Training the model...")
    model = train_model(X_train, y_train)
    logger.info("Model successfully trained.")

    logger.info("\nStep 3: Evaluating the model...")
    evaluate_model(model, X_test, y_test)
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    run_pipeline()