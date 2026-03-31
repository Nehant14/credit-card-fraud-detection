import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

from preprocess import load_data, preprocess_data


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
IMAGE_DIR = BASE_DIR / "images"

IMAGE_DIR.mkdir(exist_ok=True)


def evaluate():

    print("Loading dataset...")

    df = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Loading trained model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run train_model first or check that the models folder exists."
        )

    saved = joblib.load(MODEL_PATH)
    model = saved["classifier"]
    anomaly_model = saved["anomaly"]

    X_test["anomaly_score"] = -anomaly_model.decision_function(X_test)

    print("Making predictions...")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print("\nClassification Report:\n")

    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:\n")

    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    plt.figure(figsize=(6,4))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(IMAGE_DIR / "confusion_matrix.png")

    plt.close()

    print("\nConfusion matrix saved to images/confusion_matrix.png")

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    print("\nAverage precision (PR AUC):", avg_precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR AUC = {avg_precision:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "precision_recall_curve.png")
    plt.close()
    print("Precision-recall curve saved to images/precision_recall_curve.png")

    if len(np.unique(y_test)) > 1:
        roc = roc_auc_score(y_test, y_pred_proba)
        print("\nROC-AUC Score:", roc)
    else:
        print("\nROC-AUC cannot be calculated because only one class exists in the test set.")


if __name__ == "__main__":
    evaluate()