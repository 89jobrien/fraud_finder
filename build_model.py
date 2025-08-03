from src.preprocess import load_and_preprocess_data
from src.train import build_production_model

def main():
    """Main function to build the production model."""
    X, y, scaler = load_and_preprocess_data("data/creditcard.csv")
    build_production_model(X, y, scaler)

if __name__ == "__main__":
    main()