"""
train_model.py
--------------
Preprocesses the CitieSHealth dataset, derives a binary mental-health risk label,
trains Logistic Regression & SVM classifiers, picks the best, and persists:
  - backend/model.pkl   → trained classifier
  - backend/scaler.pkl  → fitted StandardScaler
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, 'dataset.csv')
MODEL_PATH  = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

# ── Config ────────────────────────────────────────────────────────
FEATURES       = ['age_yrs', 'sueno', 'horasfuera', 'ordenador', 'actividadfisica', 'dieta']
TARGET_COLS    = ['estres', 'bienestar', 'energia']
BINARY_COLS    = ['ordenador', 'actividadfisica', 'dieta']
YES_NO_MAP     = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}

def train():
    # ── 1. Load ───────────────────────────────────────────────────
    print("Loading dataset ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Raw shape: {df.shape}")

    # ── 2. Map Yes/No columns to 0/1 BEFORE dropna ───────────────
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(YES_NO_MAP)

    # ── 3. Select relevant columns and drop incomplete rows ────────
    df = df[FEATURES + TARGET_COLS].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    print(f"  Clean shape: {df.shape}")

    # ── 4. Derive binary risk label ───────────────────────────────
    #   "At Risk" (1) if stress > 5  OR  (wellbeing <= 4 AND energy <= 4)
    df['Risk'] = np.where(
        (df['estres'] > 5) | ((df['bienestar'] <= 4) & (df['energia'] <= 4)),
        1, 0
    )
    counts = df['Risk'].value_counts()
    print(f"  Risk distribution — 0 (No risk): {counts.get(0, 0)}, 1 (At risk): {counts.get(1, 0)}")

    if len(counts) < 2:
        raise ValueError("Only one class present in Risk label. Check the target derivation logic.")

    # ── 5. Split ──────────────────────────────────────────────────
    X = df[FEATURES]
    y = df['Risk']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 6. Scale ──────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── 7. Train both models ──────────────────────────────────────
    results = {}

    print("\nTraining Logistic Regression ...")
    lr = LogisticRegression(max_iter=500, class_weight='balanced')
    lr.fit(X_train_s, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test_s))
    print(f"  Accuracy: {lr_acc:.4f}")
    print(classification_report(y_test, lr.predict(X_test_s)))
    results['LR'] = (lr_acc, lr)

    print("Training SVM (RBF kernel) ...")
    svm = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    svm.fit(X_train_s, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test_s))
    print(f"  Accuracy: {svm_acc:.4f}")
    print(classification_report(y_test, svm.predict(X_test_s)))
    results['SVM'] = (svm_acc, svm)

    # ── 8. Pick best model ────────────────────────────────────────
    best_name, (best_acc, best_model) = max(results.items(), key=lambda kv: kv[1][0])
    print(f"\nBest model: {best_name} (accuracy={best_acc:.4f})")

    # ── 9. Persist ────────────────────────────────────────────────
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Saved model  -> {MODEL_PATH}")
    print(f"Saved scaler -> {SCALER_PATH}")
    print("Training complete!")


if __name__ == '__main__':
    train()
