"""
Spirometry Diagnosis Classifier (Keras / TensorFlow)
-----------------------------------------------------
Trains a neural network to predict DIAGNOSIS_LABEL
(Normal / Obstructive / Restrictive) and saves:
    - spirometry_model.h5     : Keras model
    - label_encoder.pkl       : LabelEncoder (class names <-> numbers)
    - scaler.pkl              : StandardScaler (REQUIRED for correct predictions)

Requirements:
    pip install pandas openpyxl scikit-learn matplotlib tensorflow h5py

Usage:
    python train_spirometry_keras.py
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------
DATA_PATH = "spirometry_training_data_500.xlsx"
MODEL_PATH = "spirometry_model.h5"
ENCODER_PATH = "label_encoder.pkl"
SCALER_PATH = "scaler.pkl"
RANDOM_STATE = 42

FEATURES = ["age", "height", "weight", "FVC", "FEV1", "PEF", "FEV1_PER_FVC"]
TARGET = "DIAGNOSIS_LABEL"

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ----------------------------------------------------------------------
# 2. Load data
# ----------------------------------------------------------------------
df = pd.read_excel(DATA_PATH).dropna(subset=FEATURES + [TARGET])
print(f"Loaded {len(df)} records")
print(df[TARGET].value_counts())

X = df[FEATURES].values
le = LabelEncoder()
y = le.fit_transform(df[TARGET])  # Normal=0, Obstructive=1, Restrictive=2
num_classes = len(le.classes_)
print("\nLabel encoding:", {c: int(i) for i, c in enumerate(le.classes_)})

# ----------------------------------------------------------------------
# 3. Split (stratified) and scale
#    Scaler is fit on training data only to avoid data leakage.
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ----------------------------------------------------------------------
# 4. Build neural network
# ----------------------------------------------------------------------
model = keras.Sequential(
    [
        layers.Input(shape=(len(FEATURES),)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ----------------------------------------------------------------------
# 5. Train (early stopping prevents overfitting)
# ----------------------------------------------------------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)
history = model.fit(
    X_train_s,
    y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1,
)

# Evaluate
y_pred = np.argmax(model.predict(X_test_s, verbose=0), axis=1)
print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(history.history["accuracy"], label="train")
axes[1].plot(history.history["val_accuracy"], label="validation")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=le.classes_, cmap="Blues", ax=ax
)
ax.set_title("Confusion Matrix - Keras Model")
plt.tight_layout()
plt.savefig("confusion_matrix_keras.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 6. Save model (.h5) and label encoder (.pkl)
# ----------------------------------------------------------------------
model.save(MODEL_PATH)  # HDF5 format
print(f"\nSaved model:         {MODEL_PATH}")

with open(ENCODER_PATH, "wb") as f:
    pickle.dump(le, f)
print(f"Saved label encoder: {ENCODER_PATH}")

# The scaler must be saved too: the network was trained on scaled inputs,
# so new data has to be scaled the same way before prediction.
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
print(f"Saved scaler:        {SCALER_PATH}")


# ----------------------------------------------------------------------
# 7. Load saved files and predict for a new patient
# ----------------------------------------------------------------------
def predict_diagnosis(age, height, weight, fvc, fev1, pef):
    """Load .h5 model + .pkl files and predict. FEV1_PER_FVC is computed automatically."""
    loaded_model = keras.models.load_model(MODEL_PATH, compile=False)
    with open(ENCODER_PATH, "rb") as f:
        loaded_le = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        loaded_scaler = pickle.load(f)

    row = np.array([[age, height, weight, fvc, fev1, pef, fev1 / fvc]])
    row_s = loaded_scaler.transform(row)
    proba = loaded_model.predict(row_s, verbose=0)[0]
    label = loaded_le.inverse_transform([int(np.argmax(proba))])[0]
    return label, {c: round(float(p), 3) for c, p in zip(loaded_le.classes_, proba)}


if __name__ == "__main__":
    print("\n--- Example prediction ---")
    label, probs = predict_diagnosis(
        age=55, height=168, weight=72, fvc=3.8, fev1=2.0, pef=4.2
    )
    print(f"Predicted diagnosis: {label}")
    print(f"Probabilities: {probs}")