import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

sys.path.append(str(SRC_DIR))

from src.train_model import train_model
from src.evaluate import evaluate
from src.visualization import plot_class_distribution, plot_correlation
from src.preprocess import load_data

print("\nTraining Model...\n")

train_model()

print("\nEvaluating Model...\n")

evaluate()

print("\nGenerating Visualizations...\n")

df = load_data()

plot_class_distribution(df)
plot_correlation(df)

print("\nImages saved inside /images folder")

print("\nProject Completed Successfully")