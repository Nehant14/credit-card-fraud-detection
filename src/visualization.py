import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_DIR = BASE_DIR / "images"

IMG_DIR.mkdir(exist_ok=True)

def plot_class_distribution(df):

    sns.countplot(x="Class", data=df)

    plt.title("Fraud vs Normal Transactions")

    plt.savefig(IMG_DIR / "class_distribution.png")

    plt.close()


def plot_correlation(df):

    plt.figure(figsize=(12,8))

    sns.heatmap(df.corr(), cmap="coolwarm")

    plt.title("Feature Correlation")

    plt.savefig(IMG_DIR / "correlation_matrix.png")

    plt.close()