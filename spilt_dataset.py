import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "traffic_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

# Drop non-feature columns
df = df.drop(columns=["ip_address", "minute", "attack_type"])

# Shuffle and split into 3 equal parts
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
indices = np.array_split(df.index, 3)

for i, idx in enumerate(indices, 1):
    split = df.loc[idx]
    out_path = os.path.join(OUTPUT_DIR, f"client_{i}.csv")
    split.to_csv(out_path, index=False)
    print(f"client_{i}.csv: {len(split)} rows")