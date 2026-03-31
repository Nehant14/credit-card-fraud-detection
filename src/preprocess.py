import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def _resample_to_ratio(X, y, ratio=20, random_state=42):
    fraud_count = y.sum()
    desired_normal_count = int(fraud_count * ratio)

    normal_samples = X[y == 0].sample(n=desired_normal_count, random_state=random_state)
    fraud_samples = X[y == 1]

    X_resampled = pd.concat([normal_samples, fraud_samples])
    y_resampled = pd.concat([
        y.loc[normal_samples.index],
        y.loc[fraud_samples.index]
    ])

    X_resampled, y_resampled = shuffle(X_resampled, y_resampled, random_state=random_state)
    return X_resampled, y_resampled


def preprocess_data(df):

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train, y_train = _resample_to_ratio(X_train, y_train, ratio=20, random_state=42)

    return X_train, X_test, y_train, y_test