"""
CrypTAS - Federated Server
Aggregates encrypted weights from all clients using FedAvg,
runs anomaly detection on weight updates, and saves global model.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from crypto.aes_utils import generate_key, load_encrypted_weights, save_encrypted_weights

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights")
GLOBAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "global_model.json")
LOGS_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "server.log")
SHARED_KEY_PASSPHRASE = "CrypTAS-Secret-2025"
NUM_CLIENTS = 3
ANOMALY_THRESHOLD = 3.0   # z-score threshold for weight anomaly detection


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)
    with open(LOGS_PATH, "a", encoding='utf-8') as f:
        f.write(line + "\n")


def load_all_client_weights():
    key = generate_key(SHARED_KEY_PASSPHRASE)
    all_weights = []
    for i in range(1, NUM_CLIENTS + 1):
        path = os.path.join(WEIGHTS_PATH, f"client_{i}_weights.json")
        if not os.path.exists(path):
            log(f"[!] Warning: weights for client {i} not found, skipping.")
            continue
        w = load_encrypted_weights(path, key)
        all_weights.append((i, w))
        log(f"[✓] Decrypted weights from client {i}")
    return all_weights


def detect_anomalies(all_weights: list) -> list:
    """
    Separable anomaly detection: flag clients whose weight update
    has a z-score > threshold compared to peers.
    Returns list of flagged client IDs.
    """
    coef_norms = []
    for (cid, w) in all_weights:
        coef_norms.append((cid, np.linalg.norm(w["coef"])))

    norms = np.array([n for _, n in coef_norms])
    if len(norms) < 2:
        return []

    mean_n, std_n = norms.mean(), norms.std()
    if std_n == 0:
        return []

    flagged = []
    for (cid, norm) in coef_norms:
        z = abs(norm - mean_n) / std_n
        if z > ANOMALY_THRESHOLD:
            log(f"[ANOMALY] Client {cid} flagged! z-score={z:.2f}, norm={norm:.4f}")
            flagged.append(cid)
        else:
            log(f"[OK] Client {cid}: z-score={z:.2f}, norm={norm:.4f}")

    return flagged


def fedavg(all_weights: list, flagged: list) -> dict:
    """FedAvg aggregation, excluding flagged clients."""
    clean = [(cid, w) for (cid, w) in all_weights if cid not in flagged]
    if not clean:
        log("[!] All clients flagged! Keeping previous global model.")
        return None

    coefs = np.stack([w["coef"] for _, w in clean])
    intercepts = np.stack([w["intercept"] for _, w in clean])
    global_weights = {
        "coef": coefs.mean(axis=0).astype(np.float32),
        "intercept": intercepts.mean(axis=0).astype(np.float32)
    }
    log(f"[✓] FedAvg aggregated {len(clean)} clients (excluded: {flagged})")
    return global_weights


def save_global_model(weights: dict):
    data = {
        "coef": weights["coef"].tolist(),
        "intercept": weights["intercept"].tolist(),
        "updated_at": datetime.now().isoformat()
    }
    with open(GLOBAL_MODEL_PATH, "w") as f:
        json.dump(data, f, indent=2)
    log(f"[✓] Global model saved to {GLOBAL_MODEL_PATH}")


def aggregate():
    log("=" * 50)
    log("Starting federated aggregation round...")
    all_weights = load_all_client_weights()

    if not all_weights:
        log("[!] No client weights found. Aborting.")
        return

    flagged = detect_anomalies(all_weights)
    global_weights = fedavg(all_weights, flagged)

    if global_weights:
        save_global_model(global_weights)

    log("Aggregation round complete.")
    log("=" * 50)

    return {
        "clients_received": len(all_weights),
        "flagged": flagged,
        "success": global_weights is not None
    }


if __name__ == "__main__":
    result = aggregate()
    print("\nResult:", result)
