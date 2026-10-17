"""
Spirometry Diagnosis Classifier
--------------------------------
Trains a machine-learning model to predict DIAGNOSIS_LABEL
(Normal / Obstructive / Restrictive) from spirometry data.

Requirements:
    pip install pandas openpyxl scikit-learn matplotlib joblib

Usage:
    python train_spirometry_model.py
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------
DATA_PATH = "spirometry_training_data_500.xlsx"
MODEL_PATH = "spirometry_model.joblib"
RANDOM_STATE = 42

FEATURES = ["age", "height", "weight", "FVC", "FEV1", "PEF", "FEV1_PER_FVC"]
TARGET = "DIAGNOSIS_LABEL"

# ----------------------------------------------------------------------
# 2. Load and inspect data
# ----------------------------------------------------------------------
df = pd.read_excel(DATA_PATH)
print(f"Loaded {len(df)} records, {df.shape[1]} columns")
print("\nClass distribution:")
print(df[TARGET].value_counts())
print("\nMissing values:", int(df.isna().sum().sum()))

# Drop any incomplete rows (none expected for this dataset)
df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
le = LabelEncoder()
y = le.fit_transform(df[TARGET])  # Normal=0, Obstructive=1, Restrictive=2
print("\nLabel encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

# ----------------------------------------------------------------------
# 3. Train / test split (stratified keeps class proportions)
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"\nTrain: {len(X_train)} records | Test: {len(X_test)} records")

# ----------------------------------------------------------------------
# 4. Compare candidate models with 5-fold cross-validation
# ----------------------------------------------------------------------
candidates = {
    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}

print("\n--- 5-fold cross-validation (training set) ---")
for name, model in candidates.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    cv_results[name] = scores.mean()
    print(f"{name:20s} accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})")

best_name = max(cv_results, key=cv_results.get)
print(f"\nBest model by CV: {best_name}")

# ----------------------------------------------------------------------
# 5. Train best model on full training set and evaluate on held-out test set
# ----------------------------------------------------------------------
best_model = candidates[best_name]
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=le.classes_, cmap="Blues", ax=ax
)
ax.set_title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# Feature importance (tree-based models only)
if hasattr(best_model, "feature_importances_"):
    importance = pd.Series(best_model.feature_importances_, index=FEATURES)
    importance = importance.sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    importance.plot(kind="barh", color="#4472C4", ax=ax)
    ax.set_title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    print("Saved: feature_importance.png")
    print("\nFeature importance:")
    print(importance.sort_values(ascending=False).round(3))

# ----------------------------------------------------------------------
# 6. Save model (with label encoder) for later use
# ----------------------------------------------------------------------
joblib.dump({"model": best_model, "label_encoder": le, "features": FEATURES}, MODEL_PATH)
print(f"\nSaved model: {MODEL_PATH}")


# ----------------------------------------------------------------------
# 7. Prediction helper for new patients
# ----------------------------------------------------------------------
def predict_diagnosis(age, height, weight, fvc, fev1, pef, model_path=MODEL_PATH):
    """Predict diagnosis for a single patient. FEV1_PER_FVC is computed automatically."""
    bundle = joblib.load(model_path)
    row = pd.DataFrame(
        [[age, height, weight, fvc, fev1, pef, fev1 / fvc]],
        columns=bundle["features"],
    )
    pred = bundle["model"].predict(row)[0]
    proba = bundle["model"].predict_proba(row)[0]
    label = bundle["label_encoder"].inverse_transform([pred])[0]
    return label, dict(zip(bundle["label_encoder"].classes_, proba.round(3)))


if __name__ == "__main__":
    print("\n--- Example prediction ---")
    label, probs = predict_diagnosis(
        age=55, height=168, weight=72, fvc=3.8, fev1=2.0, pef=4.2
    )
    print(f"Predicted diagnosis: {label}")
    print(f"Probabilities: {probs}")