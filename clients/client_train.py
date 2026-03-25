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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from crypto.aes_utils import generate_key, save_encrypted_weights, load_encrypted_weights

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "server", "weights")
SHARED_KEY_PASSPHRASE = "CrypTAS-Secret-2025"   # In production, use PKI


def load_client_data(client_id: int):
    path = os.path.join(PROCESSED_PATH, f"client_{client_id}.csv")
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values
    return X, y


def train_client(client_id: int, global_weights: dict = None):
    print(f"\n[Client {client_id}] Loading data...")
    X, y = load_client_data(client_id)

    model = LogisticRegression(max_iter=200, solver="lbfgs", random_state=42)

    # Warm-start from global model if available
    if global_weights:
        coef = global_weights.get("coef")
        intercept = global_weights.get("intercept")
        if coef is not None and intercept is not None:
            model.coef_ = coef
            model.intercept_ = intercept
            model.classes_ = np.array([0, 1])

    print(f"[Client {client_id}] Training on {len(X)} samples...")
    model.fit(X, y)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"[Client {client_id}] Local accuracy: {acc:.4f}")

    # Extract weights
    weights = {
        "coef": model.coef_.astype(np.float32),
        "intercept": model.intercept_.astype(np.float32)
    }

    # Encrypt and save
    key = generate_key(SHARED_KEY_PASSPHRASE)
    os.makedirs(WEIGHTS_PATH, exist_ok=True)
    out_path = os.path.join(WEIGHTS_PATH, f"client_{client_id}_weights.json")
    save_encrypted_weights(weights, key, out_path)

    return acc


if __name__ == "__main__":
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    train_client(client_id)
