# MAIZE LEAF DISEASE DETECTION USING RANDOM FOREST CLASSIFIER
# GROUP 27 — Amos Kibet Cheruiyot, Shaban J Shaban, Eliwa Watulevi
# University of Eastern Africa, Baraton | AI Course — Software Engineering
#
# HOW TO RUN:
#   1. pip install numpy pandas scikit-learn matplotlib seaborn pillow opencv-python
#   2. Download PlantVillage dataset: https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset
#      Extract so maize class folders are inside dataset/data/, e.g.:
#        dataset/data/Corn_(maize)___healthy/
#        dataset/data/Corn_(maize)___Northern_Leaf_Blight/
#        dataset/data/Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/
#        dataset/data/Corn_(maize)___Common_rust_/
#   3. Run: python maize_disease_detector.py

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import joblib

warnings.filterwarnings("ignore")

# CONFIGURATION
DATA_DIR = "dataset/data"
IMG_SIZE = (64, 64)
TEST_SIZE = 0.30
RANDOM_STATE = 42
N_ESTIMATORS = 150
MAX_DEPTH = None
MODEL_SAVE_PATH = "maize_rf_model.pkl"

CLASS_NAME_MAP = {
    "Blight":       "Maize Leaf Blight",
    "Common_Rust":  "Maize Rust",
    "Gray_Leaf":    "Gray Leaf Spot",
    "Healthy":      "Healthy",
}


def resolve_class_name(folder_name: str) -> str | None:
    """Map a PlantVillage folder name to one of our 4 clean class labels."""
    for key, label in CLASS_NAME_MAP.items():
        if key.lower() in folder_name.lower():
            return label
    return None


def extract_features(img_array: np.ndarray) -> np.ndarray:
    """
    Extract a feature vector from a (H, W, 3) uint8 image array.

    Features (section 2.3):
    • Colour histogram per channel (R, G, B) — 32 bins each = 96 features
    • Flattened resized pixel values normalised to [0, 1]
    • Mean and std of each channel — 6 features

    Total = 96 + (64*64*3) + 6 = 12,390 features per image
    """
    hist_features = []
    for channel in range(3):
        hist, _ = np.histogram(img_array[:, :, channel], bins=32, range=(0, 256))
        hist = hist / hist.sum()
        hist_features.extend(hist)

    pixel_features = img_array.flatten() / 255.0

    stat_features = []
    for channel in range(3):
        stat_features.append(img_array[:, :, channel].mean() / 255.0)
        stat_features.append(img_array[:, :, channel].std() / 255.0)

    return np.concatenate([hist_features, pixel_features, stat_features])


def load_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Walk data_dir, load every maize image, extract features, return (X, y, class_names)."""
    if not os.path.exists(data_dir):
        print(f"\n[ERROR] Dataset folder not found: '{data_dir}'")
        print("Please download the PlantVillage dataset from Kaggle and update DATA_DIR.")
        sys.exit(1)

    X, y = [], []
    skipped = 0
    folder_names = sorted(os.listdir(data_dir))

    print("\n" + "=" * 60)
    print("  LOADING DATASET")
    print("=" * 60)
    print(f"  Source folder : {os.path.abspath(data_dir)}")
    print(f"  Image size    : {IMG_SIZE[0]}×{IMG_SIZE[1]} px\n")

    for folder in folder_names:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        label = resolve_class_name(folder)
        if label is None:
            continue

        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        loaded = 0
        for fname in image_files:
            img_path = os.path.join(folder_path, fname)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMG_SIZE, Image.LANCZOS)
                img_array = np.array(img)
                features = extract_features(img_array)
                X.append(features)
                y.append(label)
                loaded += 1
            except Exception:
                skipped += 1

        print(f"  [{label:<22}]  {loaded:>4} images loaded  (folder: {folder})")

    print()
    if skipped:
        print(f"  Skipped {skipped} unreadable files.")

    if len(X) == 0:
        print("[ERROR] No images were loaded. Check your DATA_DIR and folder structure.")
        sys.exit(1)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    class_names = sorted(list(set(y)))

    print(f"\n  Total images loaded : {len(X)}")
    print(f"  Feature vector size : {X.shape[1]}")
    print(f"  Classes found       : {class_names}")
    return X, y, class_names


def split_data(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset 70% train / 30% test with stratification (section 2.3)."""
    print("\n" + "=" * 60)
    print("  TRAIN / TEST SPLIT  (70% / 30%)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # ensures balanced class distribution
    )

    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing  samples : {len(X_test)}\n")

    unique, counts = np.unique(y_train, return_counts=True)
    print("  Training class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"    {cls:<22} : {cnt}")

    return X_train, X_test, y_train, y_test


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train a Random Forest Classifier (section 2.3)."""
    print("\n" + "=" * 60)
    print("  TRAINING RANDOM FOREST CLASSIFIER")
    print("=" * 60)
    print(f"  n_estimators : {N_ESTIMATORS}")
    print(f"  max_depth    : {MAX_DEPTH if MAX_DEPTH else 'unlimited'}")
    print(f"  random_state : {RANDOM_STATE}")
    print("\n  Training in progress... (this may take a few minutes)\n")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",      # consider sqrt(n_features) at each split — standard RF
        class_weight="balanced",  # handles class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"  Training complete in {elapsed:.1f} seconds.")
    print(f"  Number of trees : {len(model.estimators_)}")
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict:
    """Evaluate using accuracy, precision, recall, F1, and confusion matrix (section 2.4)."""
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)

    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test,  y_pred)

    print(f"\n  Train Accuracy : {train_acc * 100:.2f}%")
    print(f"  Test  Accuracy : {test_acc  * 100:.2f}%")

    if test_acc >= 0.80:
        print("\n  ✔ Accuracy exceeds 80% — model is fit for practical use (section 2.4).")
    else:
        print("\n  ✘ Accuracy below 80% — consider tuning parameters.")

    print("\n  Classification Report (Precision / Recall / F1):")
    print("  " + "-" * 56)
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    for line in report.split("\n"):
        print("  " + line)

    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    print("\n  Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df.to_string())

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "report": report,
    }


def plot_results(
    results: dict,
    class_names: list[str],
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    save_dir: str = ".",
) -> None:
    """Generate and save four plots: confusion matrix, class distribution, feature importances, accuracy summary."""
    print("\n" + "=" * 60)
    print("  GENERATING PLOTS")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 11})

    # Plot 1: Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = results["confusion_matrix"]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Confusion Matrix — Maize Leaf Disease Classifier\n(Random Forest, GROUP 27)", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    path1 = os.path.join(save_dir, "plot_1_confusion_matrix.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path1}")

    # Plot 2: Class Distribution
    unique, counts = np.unique(y, return_counts=True)
    colors = ["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(unique, counts, color=colors[:len(unique)], edgecolor="white", linewidth=0.8)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                str(count), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Dataset Class Distribution — PlantVillage Maize Images", fontsize=13, pad=12)
    ax.set_xlabel("Disease Class", fontsize=11)
    ax.set_ylabel("Number of Images", fontsize=11)
    ax.set_xticklabels(unique, rotation=20, ha="right")
    plt.tight_layout()
    path2 = os.path.join(save_dir, "plot_2_class_distribution.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path2}")

    # Plot 3: Top 20 Feature Importances
    importances = model.feature_importances_
    top_n = 20
    indices = np.argsort(importances)[-top_n:][::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(top_n), importances[indices], color="#3498db", edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest", fontsize=13, pad=12)
    ax.set_xlabel("Feature Index (ranked)", fontsize=11)
    ax.set_ylabel("Importance Score", fontsize=11)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([f"F{i}" for i in indices], rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    path3 = os.path.join(save_dir, "plot_3_feature_importances.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path3}")

    # Plot 4: Accuracy Summary
    fig, ax = plt.subplots(figsize=(6, 4))
    labels_ = ["Train Accuracy", "Test Accuracy"]
    values  = [results["train_accuracy"] * 100, results["test_accuracy"] * 100]
    bar_colors = ["#27ae60" if v >= 80 else "#e74c3c" for v in values]
    bars = ax.bar(labels_, values, color=bar_colors, width=0.4, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.2f}%", ha="center", fontsize=13, fontweight="bold")
    ax.axhline(80, color="red", linestyle="--", linewidth=1.2, label="80% threshold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Model Accuracy — Train vs Test", fontsize=13, pad=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path4 = os.path.join(save_dir, "plot_4_accuracy_summary.png")
    plt.savefig(path4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path4}")


def predict_single_image(model: RandomForestClassifier, image_path: str) -> str:
    """
    Given a path to any maize leaf image, return the predicted disease class.

    Usage:
        result = predict_single_image(model, "my_leaf_photo.jpg")
        print(result)
    """
    if not os.path.exists(image_path):
        return f"[ERROR] File not found: {image_path}"

    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img)
    features = extract_features(img_array).reshape(1, -1)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_
    confidence = max(probabilities) * 100

    print(f"\n  Leaf image   : {image_path}")
    print(f"  Prediction   : {prediction}")
    print(f"  Confidence   : {confidence:.1f}%")
    print("\n  Class probabilities:")
    for cls, prob in sorted(zip(classes, probabilities), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"    {cls:<22} {prob*100:5.1f}%  {bar}")

    return prediction


def save_model(model: RandomForestClassifier, path: str) -> None:
    """Save trained model to disk using joblib."""
    joblib.dump(model, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\n  Model saved to: {path}  ({size_mb:.1f} MB)")


def load_model(path: str) -> RandomForestClassifier:
    """Load a previously saved model from disk."""
    model = joblib.load(path)
    print(f"  Model loaded from: {path}")
    return model


def main():
    print("\n" + "=" * 60)
    print("  MAIZE LEAF DISEASE DETECTION — GROUP 27")
    print("  Random Forest Classifier | AI Course, Baraton")
    print("=" * 60)

    X, y, class_names = load_dataset(DATA_DIR)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    results = evaluate_model(model, X_train, X_test, y_train, y_test, class_names)
    plot_results(results, class_names, model, X, y, save_dir="output_plots")
    save_model(model, MODEL_SAVE_PATH)

    # Uncomment to predict a single image:
    # predict_single_image(model, "test_leaf.jpg")

    print("\n" + "=" * 60)
    print("  RUN COMPLETE")
    print("=" * 60)
    print(f"  Test Accuracy  : {results['test_accuracy'] * 100:.2f}%")
    print(f"  Plots saved in : output_plots/")
    print(f"  Model saved as : {MODEL_SAVE_PATH}")
    if results["test_accuracy"] >= 0.80:
        print("\n  Result: Model meets the 80% accuracy target from section 2.4.")
        print("  It is ready for practical deployment in the Rift Valley region.")
    else:
        print("\n  Result: Accuracy below target. Try:")
        print("    • Increasing N_ESTIMATORS to 200 or 300")
        print("    • Increasing IMG_SIZE to (128, 128)")
        print("    • Collecting more labelled local images")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
