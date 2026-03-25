"""
CrypTAS - Main Orchestrator
Runs the full federated learning pipeline:
  1. Preprocess data
  2. Train each client
  3. Aggregate on server
  4. Print results
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clients.client_train import train_client
from server.federated_server import aggregate

NUM_ROUNDS = 3
NUM_CLIENTS = 3


def run_pipeline():
    print("\n" + "="*60)
    print("  CrypTAS - Federated Learning Pipeline")
    print("="*60)

    results = []

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n{'='*60}")
        print(f"  ROUND {round_num}/{NUM_ROUNDS}")
        print(f"{'='*60}")

        # Step 1: Train all clients
        client_accs = {}
        for cid in range(1, NUM_CLIENTS + 1):
            acc = train_client(cid)
            client_accs[cid] = acc

        # Step 2: Server aggregation + anomaly detection
        print(f"\n[Server] Running FedAvg aggregation for round {round_num}...")
        result = aggregate()

        results.append({
            "round": round_num,
            "client_accuracies": client_accs,
            "aggregation_result": result
        })

        print(f"\n[Round {round_num} Summary]")
        for cid, acc in client_accs.items():
            print(f"  Client {cid} accuracy: {acc:.4f}")
        print(f"  Flagged clients: {result.get('flagged', [])}")

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    for r in results:
        print(f"  Round {r['round']}: avg acc={sum(r['client_accuracies'].values())/NUM_CLIENTS:.4f}, flagged={r['aggregation_result'].get('flagged',[])}")

    print("\nTo view the dashboard:")
    print("  1. Run: python api.py")
    print("  2. Open: dashboard/index.html in a browser")


if __name__ == "__main__":
    run_pipeline()
