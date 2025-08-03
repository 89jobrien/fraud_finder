from sklearn.model_selection import train_test_split
from src.preprocess import get_train_test_split
from src.train import train_model
from src.evaluate import evaluate_model
from loguru import logger

def run_pipeline():
    """
    Runs the full data processing, model training, and evaluation pipeline.
    """
    data_path = "data/creditcard.csv"

    logger.info("Step 1: Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = get_train_test_split(data_path)
    logger.info("Data successfully split.")

    logger.info("\nStep 2: Training the model...")
    model = train_model(X_train, y_train)
    logger.info("Model successfully trained.")

    logger.info("\nStep 3: Evaluating the model...")
    evaluate_model(model, X_test, y_test)
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    run_pipeline()