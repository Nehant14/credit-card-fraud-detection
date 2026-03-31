import joblib
from sklearn.ensemble import GradientBoostingClassifier
from pathlib import Path
from preprocess import load_data, preprocess_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"


def train_model():

    df = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    train_model()