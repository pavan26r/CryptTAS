import os
import sys
import json
from datetime import datetime
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd   # ✅ FIXED

sys.path.insert(0, os.path.dirname(__file__))

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GLOBAL_MODEL_PATH = os.path.join(BASE_DIR, "server", "global_model.json")
LOGS_PATH = os.path.join(BASE_DIR, "logs", "server.log")
WEIGHTS_PATH = os.path.join(BASE_DIR, "server", "weights")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "traffic_dataset.csv")  # ✅ FIXED

round_history = []


# ───────────────── STATUS ─────────────────
@app.route("/api/status")
def status():
    clients_ready = sum(
        os.path.exists(os.path.join(WEIGHTS_PATH, f"client_{i}_weights.json"))
        for i in range(1, 4)
    )

    global_model_exists = os.path.exists(GLOBAL_MODEL_PATH)

    return jsonify({
        "clients_ready": clients_ready,
        "total_clients": 3,
        "global_model_ready": global_model_exists,
        "rounds_completed": len(round_history)
    })


# ───────────────── AGGREGATION ─────────────────
@app.route("/api/aggregate", methods=["POST"])
def aggregate_route():
    try:
        from server.federated_server import aggregate
        result = aggregate()

        round_history.append({
            "round": len(round_history) + 1,
            "timestamp": datetime.now().isoformat(),
            "result": result
        })

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ───────────────── ROUNDS ─────────────────
@app.route("/api/rounds")
def rounds():
    return jsonify({"rounds": round_history})


# ───────────────── LOGS ─────────────────
@app.route("/api/logs")
def logs():
    if not os.path.exists(LOGS_PATH):
        return jsonify({"logs": []})

    with open(LOGS_PATH) as f:
        lines = f.readlines()

    return jsonify({"logs": [l.strip() for l in lines[-50:]]})


# ───────────────── ANOMALIES ─────────────────
@app.route("/api/anomalies")
def anomalies():
    if not os.path.exists(DATA_PATH):
        return jsonify({"anomalies": []})

    df = pd.read_csv(DATA_PATH)
    df = df[df["label"] == 1]

    return jsonify({
        "anomalies": [
            {
                "type": row["attack_type"],
                "rpm": int(row["requests_per_min"]),
                "error": float(row["error_rate"]),
                "time": row["minute"],
                "ip": row["ip_address"]
            }
            for _, row in df.iterrows()
        ]
    })


# ───────────────── STATS ─────────────────
@app.route("/api/stats")
def stats():
    if not os.path.exists(DATA_PATH):
        return jsonify({"attack_counts": {}})

    df = pd.read_csv(DATA_PATH)

    return jsonify({
        "attack_counts": df["attack_type"].value_counts().to_dict()
    })


# ───────────────── DASHBOARD ─────────────────
@app.route("/")
def dashboard():
    return send_from_directory("dashboard", "index.html")


# ───────────────── RUN ─────────────────
if __name__ == "__main__":
    print("🚀 CrypTAS API running on http://localhost:5000")
    app.run(debug=True)