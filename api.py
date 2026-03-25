"""
CrypTAS - REST API Server
Exposes endpoints for the React dashboard to query system status,
run training rounds, and view anomaly logs.
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GLOBAL_MODEL_PATH = os.path.join(BASE_DIR, "server", "global_model.json")
LOGS_PATH = os.path.join(BASE_DIR, "logs", "server.log")
WEIGHTS_PATH = os.path.join(BASE_DIR, "server", "weights")

round_history = []   # In-memory round history (use DB for production)


@app.route("/api/status", methods=["GET"])
def status():
    clients_ready = 0
    for i in range(1, 4):
        path = os.path.join(WEIGHTS_PATH, f"client_{i}_weights.json")
        if os.path.exists(path):
            clients_ready += 1

    global_model_exists = os.path.exists(GLOBAL_MODEL_PATH)
    model_info = {}
    if global_model_exists:
        with open(GLOBAL_MODEL_PATH) as f:
            model_data = json.load(f)
        model_info["updated_at"] = model_data.get("updated_at", "unknown")

    return jsonify({
        "system": "CrypTAS",
        "clients_ready": clients_ready,
        "total_clients": 3,
        "global_model_ready": global_model_exists,
        "model_info": model_info,
        "rounds_completed": len(round_history)
    })


@app.route("/api/aggregate", methods=["POST"])
def trigger_aggregation():
    """Trigger a federated aggregation round."""
    try:
        from server.federated_server import aggregate
        result = aggregate()
        round_history.append({
            "round": len(round_history) + 1,
            "timestamp": datetime.now().isoformat(),
            "result": result
        })
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/rounds", methods=["GET"])
def get_rounds():
    return jsonify({"rounds": round_history})


@app.route("/api/logs", methods=["GET"])
def get_logs():
    if not os.path.exists(LOGS_PATH):
        return jsonify({"logs": []})
    with open(LOGS_PATH) as f:
        lines = f.readlines()
    return jsonify({"logs": [l.strip() for l in lines[-50:]]})  # last 50 lines


@app.route("/api/model", methods=["GET"])
def get_model():
    if not os.path.exists(GLOBAL_MODEL_PATH):
        return jsonify({"error": "No global model yet"}), 404
    with open(GLOBAL_MODEL_PATH) as f:
        model = json.load(f)
    # Return metadata only (not full weights)
    return jsonify({
        "updated_at": model.get("updated_at"),
        "coef_shape": [len(model["coef"]), len(model["coef"][0])],
        "intercept": model["intercept"]
    })


@app.route("/api/anomalies", methods=["GET"])
def get_anomalies():
    anomalies = []
    for r in round_history:
        flagged = r.get("result", {}).get("flagged", [])
        if flagged:
            anomalies.append({
                "round": r["round"],
                "timestamp": r["timestamp"],
                "flagged_clients": flagged
            })
    return jsonify({"anomalies": anomalies, "total": len(anomalies)})


@app.route("/")
def dashboard():
    return send_from_directory("dashboard", "index.html")


if __name__ == "__main__":
    print("[CrypTAS API] Starting on http://localhost:5000")
    app.run(debug=True, port=5000)
