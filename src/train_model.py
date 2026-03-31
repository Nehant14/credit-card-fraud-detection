import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, VotingClassifier
from xgboost import XGBClassifier
from preprocess import load_data, preprocess_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"


def train_model():

    df = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training anomaly detection model...")
    anomaly_model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.01,
        random_state=42,
        n_jobs=-1
    )
    anomaly_model.fit(X_train)

    X_train["anomaly_score"] = -anomaly_model.decision_function(X_train)
    X_test["anomaly_score"] = -anomaly_model.decision_function(X_test)

    print("Training ensemble classifier...")
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    xgb_model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    model = VotingClassifier(
        estimators=[("gb", gb_model), ("xgb", xgb_model)],
        voting="soft"
    )
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": model, "anomaly": anomaly_model}, MODEL_PATH)

    print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    train_model()