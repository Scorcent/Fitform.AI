"""
model.py — Optional Machine Learning Model (Random Forest)

This module demonstrates how a simple ML classifier could replace
(or supplement) the rule-based analysis.

Features used:
  - min_elbow_angle
  - avg_body_alignment_angle

Label:
  1 = correct push-up
  0 = incorrect push-up

A small synthetic dataset is generated for educational purposes.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ──────────────────────────────────────────────
# 1. Generate synthetic training data
# ──────────────────────────────────────────────
def generate_synthetic_data(n_samples: int = 200) -> tuple:
    """
    Create fake but realistic training data.

    These thresholds match the rule-based analyzer:
      "Correct" push-ups   → elbow ≤ 95° (deep) and alignment ≥ 160° (straight)
      "Incorrect" push-ups → anything else, with some noise
    """
    rng = np.random.default_rng(42)

    X = []
    y = []

    for _ in range(n_samples):
        if rng.random() < 0.5:
            # Generate a CORRECT example
            elbow     = rng.uniform(45, 95)      # deep enough
            alignment = rng.uniform(160, 180)    # body is straight
            label = 1
        else:
            # Generate an INCORRECT example
            elbow     = rng.uniform(70, 160)     # often too shallow
            alignment = rng.uniform(120, 175)    # often sagging
            # Make sure at least one criterion fails
            if elbow <= 95 and alignment >= 160:
                elbow = rng.uniform(100, 160)    # force failure
            label = 0

        X.append([elbow, alignment])
        y.append(label)

    return np.array(X), np.array(y)


# ──────────────────────────────────────────────
# 2. Train the Random Forest model
# ──────────────────────────────────────────────
def train_model():
    """
    Train a RandomForestClassifier on synthetic data.
    Returns the trained model and the test-set accuracy.
    """
    X, y = generate_synthetic_data(n_samples=200)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred   = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"[ML Model] Training complete — accuracy: {accuracy:.2%}")
    return clf, accuracy


# ──────────────────────────────────────────────
# 3. Predict with the trained model
# ──────────────────────────────────────────────
def predict(model, min_elbow_angle: float, avg_body_alignment: float) -> str:
    """
    Given angle measurements, predict 'correct' or 'incorrect'.
    """
    features = np.array([[min_elbow_angle, avg_body_alignment]])
    prediction = model.predict(features)[0]
    return "correct" if prediction == 1 else "incorrect"


# ──────────────────────────────────────────────
# Quick demo when run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    model, acc = train_model()
    print(f"Accuracy: {acc:.2%}")
    print("Predict (elbow=85, alignment=165):", predict(model, 85, 165))
    print("Predict (elbow=120, alignment=130):", predict(model, 120, 130))
