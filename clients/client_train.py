"""
CrypTAS - Federated Learning Client
Each client trains a local Logistic Regression model on its data slice,
then encrypts weights with AES-256 before sending to server.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from crypto.aes_utils import generate_key, save_encrypted_weights, load_encrypted_weights

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "server", "weights")
SHARED_KEY_PASSPHRASE = "CrypTAS-Secret-2025"

def load_client_data(client_id: int):
    path = os.path.join(PROCESSED_PATH, f"client_{client_id}.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"[Client {client_id}] Data file not found: {path}")

    df = pd.read_csv(path)

    # Drop any leftover non-feature columns if present
    cols_to_drop = [c for c in ["ip_address", "minute", "attack_type"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    if "label" not in df.columns:
        raise ValueError(f"[Client {client_id}] 'label' column not found in dataset.")

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(int)

    return X, y


def train_client(client_id: int, global_weights: dict = None):
    print(f"\n[Client {client_id}] Loading data...")
    X, y = load_client_data(client_id)

    # Scale features — important for Logistic Regression convergence
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = LogisticRegression(
        max_iter=500,          # increased from 200 — prevents convergence warnings
        solver="lbfgs",
        random_state=42,
        warm_start=True,
        class_weight="balanced"  # handles imbalanced Normal vs Attack classes
    )

    # Warm-start from global model if available
    if global_weights:
        coef = global_weights.get("coef")
        intercept = global_weights.get("intercept")
        if coef is not None and intercept is not None:
            try:
                model.coef_ = np.array(coef, dtype=np.float32)
                model.intercept_ = np.array(intercept, dtype=np.float32)
                model.classes_ = np.array([0, 1])
            except Exception as e:
                print(f"[Client {client_id}] Warning: Could not load global weights: {e}")

    print(f"[Client {client_id}] Training on {len(X)} samples, {X.shape[1]} features...")
    model.fit(X, y)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"[Client {client_id}] Local accuracy: {acc:.4f}")
    print(classification_report(y, preds, target_names=["Normal", "Attack"], zero_division=0))

    # Extract weights
    weights = {
        "coef": model.coef_.tolist(),       # JSON-serializable (list, not ndarray)
        "intercept": model.intercept_.tolist()
    }

    # Encrypt and save
    key = generate_key(SHARED_KEY_PASSPHRASE)
    os.makedirs(WEIGHTS_PATH, exist_ok=True)
    out_path = os.path.join(WEIGHTS_PATH, f"client_{client_id}_weights.json")
    save_encrypted_weights(weights, key, out_path)

    print(f"[Client {client_id}] Encrypted weights saved to {out_path}")
    return acc


if __name__ == "__main__":
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    train_client(client_id)