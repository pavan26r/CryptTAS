# CrypTAS — Cryptography-Enabled Federated Threat Analysis System

A lightweight federated learning system for network intrusion detection (NSL-KDD),
with AES-256 weight encryption and separable anomaly detection.

---

## Project Structure

```
CrypTAS/
├── data/
│   ├── raw/              ← Put KDDTrain+.txt here
│   └── processed/        ← Generated client CSVs go here
├── clients/
│   ├── preprocess.py     ← Data preprocessing
│   └── client_train.py   ← Local training + AES encryption
├── server/
│   ├── federated_server.py  ← FedAvg + anomaly detection
│   └── weights/          ← Encrypted weight JSONs
├── crypto/
│   └── aes_utils.py      ← AES-256 CBC encrypt/decrypt
├── dashboard/
│   └── index.html        ← React dashboard (no build needed)
├── logs/
│   └── server.log        ← Auto-generated logs
├── api.py                ← Flask REST API for dashboard
├── run_pipeline.py       ← Full pipeline orchestrator
└── requirements.txt
```

---

## Step-by-Step Setup & Run

### Step 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### Step 2 — Download NSL-KDD Dataset

1. Go to: https://www.kaggle.com/datasets/hassan06/nslkdd
2. Download `KDDTrain+.txt`
3. Place it in: `data/raw/KDDTrain+.txt`

---

### Step 3 — Preprocess the data

```bash
cd clients
python preprocess.py
cd ..
```

This will create:
- `data/processed/client_1.csv`
- `data/processed/client_2.csv`
- `data/processed/client_3.csv`

---

### Step 4 — Run the full pipeline (training + aggregation)

```bash
python run_pipeline.py
```

This will:
1. Train each client locally on its data slice
2. Encrypt weights with AES-256
3. Server decrypts, runs anomaly detection (z-score)
4. FedAvg aggregation → saves global model
5. Repeats for 3 rounds

---

### Step 5 — Launch the API server

In a new terminal:
```bash
python api.py
```

API runs on: http://localhost:5000

---

### Step 6 — Open the Dashboard

Open `dashboard/index.html` in your browser.

The dashboard shows:
- Client status (how many have submitted weights)
- Global model status
- Round history with anomaly flags
- Live server logs
- "Run Aggregation" button to trigger a round

---

## Individual Commands

Train only one specific client:
```bash
cd clients
python client_train.py 1   # trains client 1
python client_train.py 2   # trains client 2
python client_train.py 3   # trains client 3
```

Run only the server aggregation:
```bash
cd server
python federated_server.py
```

Test AES encryption:
```bash
cd crypto
python aes_utils.py
```

---

## How It Works

| Component | Description |
|-----------|-------------|
| **AES-256-CBC** | Each client encrypts its model weights before transmission |
| **FedAvg** | Server averages clean client weights into a global model |
| **Anomaly Detection** | Z-score on weight norms; clients > 3σ are flagged and excluded |
| **NSL-KDD** | 41-feature network traffic dataset; label binarized (normal vs attack) |
| **Logistic Regression** | Lightweight binary classifier; warm-started from global model each round |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | System status |
| POST | `/api/aggregate` | Trigger aggregation round |
| GET | `/api/rounds` | Round history |
| GET | `/api/logs` | Server logs (last 50 lines) |
| GET | `/api/model` | Global model metadata |
| GET | `/api/anomalies` | Flagged anomaly events |

---

## Requirements

- Python 3.9+
- pip packages: see `requirements.txt`
- NSL-KDD dataset (free on Kaggle)
- Modern browser for dashboard

---

## Next Phases (Planned)

- [ ] Phase 4: Deep learning model (MLP/LSTM)
- [ ] Phase 5: Differential privacy (noise injection)
- [ ] Phase 6: Multi-round convergence tracking
- [ ] Phase 7: Real-time threat scoring
