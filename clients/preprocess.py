"""
CrypTAS Phase 2 - NSL-KDD Preprocessing
Loads KDDTest+.txt, encodes, scales, and splits into 3 client CSVs.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Column names for NSL-KDD
COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

CATEGORICAL = ["protocol_type", "service", "flag"]
RAW_PATH = "../data/raw/KDDTest+.txt"
PROCESSED_PATH = "../data/processed"
NUM_CLIENTS = 3

def preprocess():
    print("[*] Loading NSL-KDD dataset...")
    df = pd.read_csv(RAW_PATH, header=None, names=COLUMNS)
    df.drop(columns=["difficulty"], inplace=True)

    print("[*] Encoding categorical columns...")
    for col in CATEGORICAL:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    print("[*] Binarizing labels (0=normal, 1=attack)...")
    df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

    print("[*] Scaling features with MinMaxScaler...")
    features = [c for c in df.columns if c != "label"]
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    os.makedirs(PROCESSED_PATH, exist_ok=True)

    print(f"[*] Splitting into {NUM_CLIENTS} client slices...")
    indices = np.array_split(df.index, NUM_CLIENTS)
    splits = [df.loc[idx] for idx in indices]
    for i, split in enumerate(splits):
        path = os.path.join(PROCESSED_PATH, f"client_{i+1}.csv")
        split.to_csv(path, index=False)
        print(f"    -> Saved {path} ({len(split)} rows)")

    print("[✓] Preprocessing complete!")

if __name__ == "__main__":
    preprocess()
