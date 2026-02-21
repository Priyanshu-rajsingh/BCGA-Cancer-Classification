"""
preprocess.py
-------------
Data preprocessing pipeline for BCGA Cancer Classification.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw data from a CSV file."""
    return pd.read_csv(filepath)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values."""
    return df.dropna()


def encode_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Encode categorical labels as integer codes."""
    df = df.copy()
    df[label_col] = df[label_col].astype("category").cat.codes
    return df


def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """Standardise features using training statistics."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess(filepath: str, label_col: str = "label", test_size: float = 0.2, random_state: int = 42):
    """Full preprocessing pipeline. Returns train/test splits."""
    df = load_data(filepath)
    df = handle_missing_values(df)
    df = encode_labels(df, label_col)

    X = df.drop(columns=[label_col]).values
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test, scaler = scale_features(X_train, X_test)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <data_filepath>")
        sys.exit(1)

    X_train, X_test, y_train, y_test, _ = preprocess(sys.argv[1])
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)
    print("Preprocessed data saved to data/processed/")
