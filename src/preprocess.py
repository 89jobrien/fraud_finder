import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new features from existing ones."""
    df['log_amount'] = np.log1p(df['Amount'])
    seconds_in_day = 24 * 60 * 60
    df['sin_time'] = np.sin(2 * np.pi * df['Time'] / seconds_in_day)
    df['cos_time'] = np.cos(2 * np.pi * df['Time'] / seconds_in_day)
    df = df.drop(['Time', 'Amount'], axis=1)
    return df

def get_processed_data(file_path: str):
    """
    Loads, engineers, and scales the full dataset.
    Returns: The full X, y, and the fitted scaler.
    """
    df = pd.read_csv(file_path)
    df = _engineer_features(df)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X, y, scaler

def get_train_test_split(file_path: str):
    """
    Processes the data and returns a train/test split.
    """
    X, y, _ = get_processed_data(file_path)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test
