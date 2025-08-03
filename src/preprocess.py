import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# imblearn is no longer needed in this file

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new features from existing ones."""
    df['log_amount'] = np.log1p(df['Amount'])
    seconds_in_day = 24 * 60 * 60
    df['sin_time'] = np.sin(2 * np.pi * df['Time'] / seconds_in_day)
    df['cos_time'] = np.cos(2 * np.pi * df['Time'] / seconds_in_day)
    df = df.drop(['Time', 'Amount'], axis=1)
    return df

def load_and_preprocess_data(file_path: str):
    """
    Loads, engineers, scales, and splits the data.
    Does NOT perform any sampling.
    """
    df = pd.read_csv(file_path)
    df = engineer_features(df)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # Split and return the original, imbalanced data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

# def load_and_preprocess_data(file_path: str):
#     """
#     Loads data, engineers features, scales data, splits into train/test sets,
#     and then applies undersampling only to the training data.
#     """
#     df = pd.read_csv(file_path)
#     df = engineer_features(df)
    
#     X = df.drop('Class', axis=1)
#     y = df['Class']
    
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     X = pd.DataFrame(X_scaled, columns=X.columns)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     smote = SMOTE(random_state=42)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)  # type: ignore

#     return X_train_resampled, X_test, y_train_resampled, y_test