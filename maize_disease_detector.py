# =============================================================================
# MAIZE LEAF DISEASE DETECTION USING RANDOM FOREST CLASSIFIER
# GROUP 27 — Amos Kibet Cheruiyot, Shaban J Shaban, Eliwa Watulevi
# University of Eastern Africa, Baraton
# AI Course — Software Engineering
# =============================================================================
#
# HOW TO RUN:
#   1. Install dependencies:
#        pip install numpy pandas scikit-learn matplotlib seaborn pillow opencv-python
#
#   2. Download the PlantVillage dataset from Kaggle:
#        https://www.kaggle.com/datasets/emmarex/plantdisease
#      Extract it so that maize/corn class folders are inside a folder, e.g.:
#        dataset/
#          Corn_(maize)___healthy/
#          Corn_(maize)___Northern_Leaf_Blight/
#          Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/
#          Corn_(maize)___Common_rust_/
#
#   3. Set DATA_DIR below to point to that folder, then run:
#        python maize_disease_detector.py
# =============================================================================

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

# =============================================================================
# CONFIGURATION — edit these as needed
# =============================================================================

DATA_DIR = "dataset/data"          # Path to folder containing the 4 maize class folders
IMG_SIZE = (64, 64)           # Image resize target (width, height)
TEST_SIZE = 0.30              # 30% test, 70% train (as per project document)
RANDOM_STATE = 42
N_ESTIMATORS = 150            # Number of trees in the forest
MAX_DEPTH = None              # None = trees grow fully; set e.g. 20 to limit depth
MODEL_SAVE_PATH = "maize_rf_model.pkl"  # Trained model will be saved here

# Mapping: substring in folder name → clean label
CLASS_NAME_MAP = {
    "Blight":       "Maize Leaf Blight",
    "Common_Rust":  "Maize Rust",
    "Gray_Leaf":    "Gray Leaf Spot",
    "Healthy":      "Healthy",
}

# =============================================================================
# SECTION 1: DATA LOADING & PREPROCESSING
# =============================================================================

def resolve_class_name(folder_name: str) -> str | None:
    """Map a PlantVillage folder name to one of our 4 clean class labels."""
    for key, label in CLASS_NAME_MAP.items():
        if key.lower() in folder_name.lower():
            return label
    return None  # Not a maize folder — skip it


def extract_features(img_array: np.ndarray) -> np.ndarray:
    """
    Extract a feature vector from a (H, W, 3) uint8 image array.

    Features used (as described in section 2.3 of the project document):
      • Colour histogram per channel (R, G, B) — 32 bins each = 96 features
        Captures colour distribution (yellowing, browning, green health)
      • Flattened resized pixel values normalised to [0, 1]
        Captures spatial texture and shape patterns
      • Mean and std of each channel — 6 features
        Compact summary of overall colour balance

    Total features = 96 + (64*64*3) + 6 = 12,390 per image
    """
    # --- Colour histograms (captures disease colour signatures) ---
    hist_features = []
    for channel in range(3):  # R, G, B
        hist, _ = np.histogram(img_array[:, :, channel], bins=32, range=(0, 256))
        hist = hist / hist.sum()  # normalise to probabilities
        hist_features.extend(hist)

    # --- Normalised pixel values (spatial/texture information) ---
    pixel_features = img_array.flatten() / 255.0

    # --- Per-channel mean and std ---
    stat_features = []
    for channel in range(3):
        stat_features.append(img_array[:, :, channel].mean() / 255.0)
        stat_features.append(img_array[:, :, channel].std() / 255.0)

    return np.concatenate([hist_features, pixel_features, stat_features])


def load_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Walk data_dir, load every image from maize class folders,
    extract features, and return (X, y_labels, class_names).
    """
    if not os.path.exists(data_dir):
        print(f"\n[ERROR] Dataset folder not found: '{data_dir}'")
        print("Please download the PlantVillage dataset from Kaggle and update DATA_DIR.")
        sys.exit(1)

    X, y = [], []
    skipped = 0
    folder_names = sorted(os.listdir(data_dir))

    print("\n" + "=" * 60)
    print("  SECTION 1: LOADING DATASET")
    print("=" * 60)
    print(f"  Source folder : {os.path.abspath(data_dir)}")
    print(f"  Image size    : {IMG_SIZE[0]}×{IMG_SIZE[1]} px")
    print()

    for folder in folder_names:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        label = resolve_class_name(folder)
        if label is None:
            continue  # not one of our 4 maize classes

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


# =============================================================================
# SECTION 2: TRAIN / TEST SPLIT
# =============================================================================

def split_data(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset 70% train / 30% test with stratification so each
    class is proportionally represented in both sets (section 2.3).
    """
    print("\n" + "=" * 60)
    print("  SECTION 2: TRAIN / TEST SPLIT  (70% / 30%)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,        # ensures balanced class distribution
    )

    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing  samples : {len(X_test)}")
    print()

    # Show per-class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print("  Training class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"    {cls:<22} : {cnt}")

    return X_train, X_test, y_train, y_test


# =============================================================================
# SECTION 3: TRAIN THE RANDOM FOREST MODEL
# =============================================================================

def train_model(
    X_train: np.ndarray, y_train: np.ndarray
) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier as described in section 2.3.

    Random Forest builds multiple decision trees on random subsets of data
    and uses majority voting among all trees for the final prediction.
    This makes it robust and handles complex image features well.
    """
    print("\n" + "=" * 60)
    print("  SECTION 3: TRAINING RANDOM FOREST CLASSIFIER")
    print("=" * 60)
    print(f"  n_estimators (trees) : {N_ESTIMATORS}")
    print(f"  max_depth            : {MAX_DEPTH if MAX_DEPTH else 'unlimited'}")
    print(f"  random_state         : {RANDOM_STATE}")
    print("\n  Training in progress... (this may take a few minutes)\n")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",   # at each split, consider sqrt(n_features) — standard RF
        class_weight="balanced",  # handles any class imbalance in the dataset
        random_state=RANDOM_STATE,
        n_jobs=-1,             # use all CPU cores
        verbose=0,
    )

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"  Training complete in {elapsed:.1f} seconds.")
    print(f"  Number of trees     : {len(model.estimators_)}")

    return model


# =============================================================================
# SECTION 4: EVALUATE THE MODEL (section 2.4 metrics)
# =============================================================================

def evaluate_model(
    model: RandomForestClassifier,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Evaluate model performance using the metrics specified in section 2.4:
      - Accuracy
      - Precision, Recall, F1-score
      - Confusion Matrix
    """
    print("\n" + "=" * 60)
    print("  SECTION 4: MODEL EVALUATION")
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
        print("\n  ✘ Accuracy below 80% — consider tuning parameters (see SECTION 5).")

    print("\n  Classification Report (Precision / Recall / F1):")
    print("  " + "-" * 56)
    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        digits=4,
    )
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


# =============================================================================
# SECTION 5: VISUALISATIONS
# =============================================================================

def plot_results(
    results: dict,
    class_names: list[str],
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    save_dir: str = ".",
) -> None:
    """
    Generate and save four plots:
      1. Confusion matrix heatmap
      2. Class distribution bar chart
      3. Feature importances (top 20)
      4. Train vs test accuracy summary
    """
    print("\n" + "=" * 60)
    print("  SECTION 5: GENERATING PLOTS")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 11})

    # ---- Plot 1: Confusion Matrix ----
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = results["confusion_matrix"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
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

    # ---- Plot 2: Class Distribution ----
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

    # ---- Plot 3: Top 20 Feature Importances ----
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

    # ---- Plot 4: Accuracy Summary ----
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


# =============================================================================
# SECTION 6: PREDICT A SINGLE LEAF IMAGE
# =============================================================================

def predict_single_image(model: RandomForestClassifier, image_path: str) -> str:
    """
    Given a path to any maize leaf image, return the predicted disease class.
    This is what a farmer would use in production.

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


# =============================================================================
# SECTION 7: SAVE AND LOAD MODEL
# =============================================================================

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


# =============================================================================
# MAIN — runs all sections in order
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  MAIZE LEAF DISEASE DETECTION — GROUP 27")
    print("  Random Forest Classifier | AI Course, Baraton")
    print("=" * 60)

    # SECTION 1 — Load data
    X, y, class_names = load_dataset(DATA_DIR)

    # SECTION 2 — Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # SECTION 3 — Train
    model = train_model(X_train, y_train)

    # SECTION 4 — Evaluate
    results = evaluate_model(model, X_train, X_test, y_train, y_test, class_names)

    # SECTION 5 — Plots
    plot_results(results, class_names, model, X, y, save_dir="output_plots")

    # SECTION 6 — Save model
    save_model(model, MODEL_SAVE_PATH)

    # ---- Demo: predict a single image (comment out if not needed) ----
    # Uncomment and point to any maize leaf image you have:
    # predict_single_image(model, "test_leaf.jpg")

    # ---- Final summary ----
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
