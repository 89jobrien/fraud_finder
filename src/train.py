import os
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from loguru import logger

# def train_model(X_train, y_train):
#     """
#     Performs a fine-grained hyperparameter search with GridSearchCV.
#     """
#     model = LGBMClassifier(random_state=42)

#     # Define a smaller, more focused grid based on prior results from a random search.
#     # For example, let's say your random search found that n_estimators around 300
#     # and learning_rate around 0.1 were best.
#     param_grid = {
#         'n_estimators': [250, 300, 350],
#         'learning_rate': [0.06, 0.1, 0.12],
#         'num_leaves': [30, 45, 50],
#     }

#     # Use GridSearchCV for an exhaustive search of the smaller grid
#     grid_search = GridSearchCV(
#         model,  # type:ignore
#         param_grid=param_grid,
#         cv=3,
#         scoring='average_precision',
#         n_jobs=-1,
#         return_train_score=True
#     )
    
#     logger.info("Starting fine-grained GridSearch...")
#     grid_search.fit(X_train, y_train)
    
#     logger.info(f"Best parameters found: {grid_search.best_params_}")
#     logger.info(f"Best AUPRC score: {grid_search.best_score_:.4f}")
    
#     return grid_search.best_estimator_

def train_model(X_train, y_train):
    """
    Performs hyperparameter tuning with RandomizedSearchCV to find the best
    LGBMClassifier model.
    """
    model = LGBMClassifier(random_state=42, class_weight='balanced')

    param_dist = {
        'n_estimators': sp_randint(100, 500),
        'learning_rate': sp_uniform(0.05, 0.2),
        'num_leaves': sp_randint(20, 100),
        'max_depth': sp_randint(5, 20),
    }

    rand_search = RandomizedSearchCV(
        model,  # type:ignore
        param_distributions=param_dist,
        n_iter=50,  # Increased from 10 to 50 for a more thorough search
        cv=5,
        scoring='average_precision',
        random_state=42,
        n_jobs=-1,
        refit=True,
        return_train_score=True
    )
    
    logger.info("Starting hyperparameter search...")
    rand_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters found: {rand_search.best_params_}")
    logger.info(f"Best AUPRC score: {rand_search.best_score_:.4f}")
    
    return rand_search.best_estimator_

def build_production_model(X, y, scaler):
    """Trains a final model on all data and saves the artifacts."""
    logger.info("Training final production model with class weighting...")
    
    # Use all the data for the final training run
    model = LGBMClassifier(random_state=42, class_weight='balanced')
    model.fit(X, y)
    logger.info("Production model training complete.")

    os.makedirs("models", exist_ok=True)
    logger.info("Saving production model and scaler to 'models/' directory...")
    joblib.dump(model, "models/fraud_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    logger.info("Artifacts saved successfully.")