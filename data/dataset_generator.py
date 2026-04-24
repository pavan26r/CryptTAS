import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
random.seed(42)
np.random.seed(42)

# ─── Config ─────────────────────────────────────────────
TOTAL_ROWS      = 5000
ATTACK_RATIO    = 0.35   # 35% attack traffic
OUTPUT_FILE     = os.path.join(BASE_DIR, "raw", "traffic_dataset.csv")
PROCESSED_DIR   = os.path.join(BASE_DIR, "processed")
NUM_CLIENTS     = 3

ATTACK_TYPES = {
    "DDoS":       0.40,   # high RPM, low unique endpoints
    "BruteForce": 0.35,   # high error rate, many login attempts
    "Scanning":   0.25,   # many unique endpoints, moderate RPM
}
# ────────────────────────────────────────────────────────


def random_ip():
    return f"192.168.{random.randint(1, 10)}.{random.randint(1, 254)}"


def random_minute(base: datetime, offset_minutes: int) -> str:
    t = base + timedelta(minutes=offset_minutes)
    return t.strftime("%Y-%m-%d %H:%M")


def generate_normal(ip, minute):
    return {
        "ip_address":        ip,
        "minute":            minute,
        "requests_per_min":  random.randint(5, 60),
        "unique_endpoints":  random.randint(2, 15),
        "error_rate":        round(random.uniform(0.0, 0.10), 3),
        "avg_response_size": random.randint(200, 1500),
        "burst_rate":        random.randint(1, 10),
        "login_attempts":    random.randint(0, 2),
        "protocol_http":     1,
        "same_endpoint_ratio": round(random.uniform(0.0, 0.3), 3),
        "attack_type":       "Normal",
        "label":             0,
    }


def generate_ddos(ip, minute):
    """High RPM, low endpoint diversity, medium error rate."""
    return {
        "ip_address":        ip,
        "minute":            minute,
        "requests_per_min":  random.randint(300, 1200),
        "unique_endpoints":  random.randint(1, 4),
        "error_rate":        round(random.uniform(0.05, 0.30), 3),
        "avg_response_size": random.randint(100, 400),
        "burst_rate":        random.randint(80, 200),
        "login_attempts":    0,
        "protocol_http":     1,
        "same_endpoint_ratio": round(random.uniform(0.7, 1.0), 3),
        "attack_type":       "DDoS",
        "label":             1,
    }


def generate_bruteforce(ip, minute):
    """Moderate RPM, very high error rate, many login attempts."""
    return {
        "ip_address":        ip,
        "minute":            minute,
        "requests_per_min":  random.randint(50, 200),
        "unique_endpoints":  random.randint(1, 3),
        "error_rate":        round(random.uniform(0.40, 0.95), 3),
        "avg_response_size": random.randint(150, 600),
        "burst_rate":        random.randint(20, 60),
        "login_attempts":    random.randint(30, 150),
        "protocol_http":     1,
        "same_endpoint_ratio": round(random.uniform(0.6, 1.0), 3),
        "attack_type":       "BruteForce",
        "label":             1,
    }


def generate_scanning(ip, minute):
    """Moderate RPM, very high unique endpoints, spread across ports."""
    return {
        "ip_address":        ip,
        "minute":            minute,
        "requests_per_min":  random.randint(40, 180),
        "unique_endpoints":  random.randint(20, 80),
        "error_rate":        round(random.uniform(0.10, 0.40), 3),
        "avg_response_size": random.randint(300, 2000),
        "burst_rate":        random.randint(10, 30),
        "login_attempts":    random.randint(0, 5),
        "protocol_http":     random.randint(0, 1),
        "same_endpoint_ratio": round(random.uniform(0.0, 0.2), 3),
        "attack_type":       "Scanning",
        "label":             1,
    }


ATTACK_GENERATORS = {
    "DDoS":       generate_ddos,
    "BruteForce": generate_bruteforce,
    "Scanning":   generate_scanning,
}


def pick_attack_type() -> str:
    types  = list(ATTACK_TYPES.keys())
    weights = list(ATTACK_TYPES.values())
    return random.choices(types, weights=weights, k=1)[0]


def build_dataset() -> pd.DataFrame:
    base_time = datetime(2026, 4, 2, 8, 0, 0)
    rows = []

    n_attacks = int(TOTAL_ROWS * ATTACK_RATIO)
    n_normal  = TOTAL_ROWS - n_attacks

    for i in range(n_normal):
        rows.append(generate_normal(random_ip(), random_minute(base_time, i % 60)))

    for i in range(n_attacks):
        atype = pick_attack_type()
        gen   = ATTACK_GENERATORS[atype]
        rows.append(gen(random_ip(), random_minute(base_time, i % 60)))

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


def add_noise(df: pd.DataFrame, noise_pct: float = 0.02) -> pd.DataFrame:
    """Flip a small % of labels to simulate real-world noise."""
    n_flip = int(len(df) * noise_pct)
    flip_idx = df.sample(n=n_flip, random_state=99).index
    df.loc[flip_idx, "label"] = 1 - df.loc[flip_idx, "label"]
    return df


def print_summary(df: pd.DataFrame):
    print("=" * 50)
    print(f"Dataset shape : {df.shape}")
    print(f"Normal rows   : {(df['label'] == 0).sum()}")
    print(f"Attack rows   : {(df['label'] == 1).sum()}")
    print("\nAttack type breakdown:")
    print(df[df['label'] == 1]['attack_type'].value_counts().to_string())
    print("\nFeature stats (numeric):")
    print(df.describe().round(2).to_string())
    print("=" * 50)


if __name__ == "__main__":
    print("Generating dataset...")
    df = build_dataset()
    df = add_noise(df)

    print_summary(df)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved raw dataset → {OUTPUT_FILE}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    indices = np.array_split(df.index, NUM_CLIENTS)
    for i, split in enumerate([df.loc[idx] for idx in indices]):
        client_path = os.path.join(PROCESSED_DIR, f"client_{i+1}.csv")
        split.to_csv(client_path, index=False)
        print(f"Saved client split → {client_path} ({len(split)} rows)")

    # ── Quick sanity: train a Random Forest ──────────────
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report

        features = [
            "requests_per_min", "unique_endpoints",
            "error_rate", "avg_response_size",
            "burst_rate", "login_attempts",
            "protocol_http", "same_endpoint_ratio",
        ]

        X = df[features]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        print("\nClassification report on test set:")
        print(classification_report(y_test, model.predict(X_test)))

        importances = pd.Series(
            model.feature_importances_, index=features
        ).sort_values(ascending=False)
        print("Feature importances:")
        print(importances.round(4).to_string())

    except ImportError:
        print("\nsklearn not installed — skipping sanity check.")
        print("Run: pip install scikit-learn")