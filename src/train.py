"""
train.py
--------
Model training and evaluation pipeline for BCGA Cancer Classification.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_processed_data(data_dir: str):
    """Load preprocessed numpy arrays from *data_dir*."""
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def train_model(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return a metrics dictionary."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    if len(np.unique(y_test)) == 2:
        metrics["auc_roc"] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return metrics, y_pred


def save_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, output_path: str) -> None:
    """Save confusion matrix plot to *output_path*."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, colorbar=True)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def save_metrics(metrics: dict, output_path: str) -> None:
    """Save evaluation metrics to a text file."""
    with open(output_path, "w") as f:
        f.write("Model Evaluation Metrics\n")
        f.write("========================\n")
        for key, value in metrics.items():
            f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")


def run(data_dir: str = "../data/processed", results_dir: str = RESULTS_DIR) -> None:
    """End-to-end training and evaluation run."""
    os.makedirs(results_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_processed_data(data_dir)

    model = train_model(X_train, y_train, n_estimators=100, random_state=42)

    metrics, y_pred = evaluate_model(model, X_test, y_test)

    save_confusion_matrix(y_test, y_pred, os.path.join(results_dir, "confusion_matrix.png"))
    save_metrics(metrics, os.path.join(results_dir, "metrics.txt"))

    print("Training complete. Results saved to", results_dir)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/processed"
    run(data_dir=data_dir)
