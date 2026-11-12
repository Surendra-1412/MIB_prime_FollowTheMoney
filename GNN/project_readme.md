Real-Time Explainable Fintech Fraud Detection Pipeline

System Architecture

+-----------------------------------------------------------------------------------+
|                          LIVE INCOMING TRANSACTION STREAM                         |
|     { raw_event_id, merchant_id, payer_id, raw_amount, mcc_raw, geo_coords }      |
+-----------------------------------------+-----------------------------------------+
                                          |
          +-------------------------------+-------------------------------+
          |                               |                               |
          v                               v                               v
+-------------------+           +-------------------+           +-------------------+
|     PILLAR 1      |           |     PILLAR 2      |           |     PILLAR 3      |
|  Spatial Engine   |           |  Temporal Engine  |           | Contextual & Geo  |
|  (Dynamic GNN)    |           |   (Transformer)   |           |     Telemetry     |
+---------+---------+           +---------+---------+           +---------+---------+
          |                               |                               |
   64D Embedding &                 Anomaly Score &                 Adaptive Velocity,
   Syndicate Score                  Sequence MSE                   MCC Surge & Geo
          |                               |                               |
          +-------------------------------+-------------------------------+
                                          |
                                          v
                        +-----------------------------------+
                        |    CatBoost Classifier (Fusion)   |
                        +-----------------+-----------------+
                                          |
                                          v
                        +-----------------------------------+
                        |     SHAP Explainability Module    |
                        +-----------------+-----------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     TERMINAL VERDICT & PERSISTENT LEDGER                          |
|      { overall_fraud_score, syndicate_score, anomaly_score, velocity_score }      |
+-----------------------------------------------------------------------------------+


Technical Executive Summary

This repository contains a standalone, real-time fraud detection engine engineered for payment gateways and merchant acquiring platforms. Designed for direct terminal execution and modular integration into existing web applications or streaming pipelines, the architecture resolves the trade-off between execution latency, seasonal adaptability, and model interpretability.

Rather than evaluating transactions in isolation, the engine analyzes every live event across three specialized intelligence pillars:

Spatial Intelligence (Dynamic GraphSAGE GNN): Maintains a live bipartite merchant-payer graph in memory. It automatically initializes new 64-dimensional graph nodes for previously unseen merchants, propagates risk across shared payers via inductive neighbor aggregation, and persists learned embeddings to disk.

Temporal Intelligence (Transformer Autoencoder): Evaluates rolling 10-transaction amount and cadence sequences via multi-head self-attention, computing a Mean Squared Error (MSE) reconstruction score normalized into an explicit Anomaly Transaction Score.

Contextual, Seasonal & Geospatial Fusion (CatBoost + Telemetry): Combines spatial embeddings, sequence scores, Haversine geospatial distance (impossible-travel terminal cloning detection), and self-calibrating Merchant Category Code (MCC) macro-surge baselines to distinguish legitimate seasonal rushes (e.g., Diwali, sports seasons) from isolated syndicate anomalies.

The engine integrates Shapley Additive Explanations (SHAP) to decompose every risk decision into pillar-level marginal contributions, outputting human-readable diagnostic reports directly in the terminal alongside an append-only JSONL audit ledger.

Data Pipeline and Processing Walkthrough

[Live Website / Gateway Stream (JSONL / Direct Call / HTTP)]
       │
       ├──> Geo & Terminal Tracker ────> Haversine Distance & Cloning Speed Check
       ├──> Adaptive MCC Tracker ──────> Self-Calibrating Seasonal Surge Multiplier
       ├──> Rolling Merchant Buffer ───> 1-Hr Velocity & Unique-Payer Diversity Ratio
       ├──> Dynamic Merchant Graph ────> Node Creation, Neighbor Aggregation & Syndicate Score
       └──> Transformer Autoencoder ───> 10-Step Sequence Reconstruction Error (Anomaly Score)
                                                    │
                                                    v
                                      [CatBoost Decision Tree Matrix]
                                                    │
                                                    v
                                      [Overall Fraud Probability Score]
                                                    │
                                                    v
                                      [SHAP Pillar Attribution Analysis]
                                                    │
                                                    v
                             [Terminal Report & Persistent JSONL Ledger]


1. Live Stream Ingestion

fraud_engine.py captures live transactions in real time without requiring an external web server, while live_website_generator.py simulates a local website checkout backend streaming live normal payments, cloned POS terminal jumps, and isolated syndicate bursts.

Standard Transaction Payload:

{
  "raw_event_id": "PAY_99812",
  "timestamp": "2026-11-12T18:23:00Z",
  "merchant_id": "M_SPORTS_HUB_99",
  "payer_id": "MULE_RING_ALPHA",
  "raw_amount": 2500000.0,
  "mcc_raw": 5941,
  "terminal_hardware_type": "WEB_REST_API",
  "terminal_id": "TID_WEB_999",
  "merchant_lat": 17.4401,
  "merchant_lon": 78.3489,
  "payer_lat": 19.0760,
  "payer_lon": 72.8777
}


2. Feature Extraction and Online Graph Learning

Dynamic Graph Node Creation & Propagation: If merchant_id is unseen, DynamicMerchantGraph initializes a 64-element floating-point vector, links the payer_id, aggregates neighboring merchant embeddings sharing that payer, and computes a normalized Syndicate Score ($S_{syn} \in [0, 1]$).

Self-Calibrating Seasonal & Diversity Filtering: AdaptiveCategoryContextTracker monitors platform-wide transaction density per mcc_raw. Broad, multi-merchant surges with high unique-payer diversity automatically scale expected velocity thresholds, whereas isolated single-merchant spikes receive no macro-cushion.

Geospatial Telemetry: TerminalLocationTracker computes the Haversine distance between payer and merchant coordinates and flags physical terminal cloning if consecutive transactions imply travel speeds exceeding $1,000\text{ km/h}$.

Sequence Transformation: The merchant's trailing 10 transactions are scaled and passed through the PyTorch Transformer Autoencoder (transformer_temporal_model.pth) to derive the Anomaly Transaction Score ($S_{anom} \in [0, 1]$).

3. Model Inference and SHAP Pillar Mapping

CatBoost (catboost_master_fraud_model.cbm) evaluates the concatenated 73-feature vector (64 GNN dimensions + 9 contextual/temporal attributes) to output the Overall Fraud Score ($P \in [0, 1]$).

SHAP marginal contributions are aggregated across the 64 GNN dimensions (Syndicate Pillar), sequence MSE (Anomaly Pillar), and transaction rate (Velocity Pillar), then printed to the terminal and appended to transactions_ledger.jsonl.

Example Structured Output Record:

{
  "event_id": "PAY_99812",
  "timestamp": "2026-11-12T18:23:00Z",
  "merchant_id": "M_SPORTS_HUB_99",
  "is_new_merchant_node": true,
  "amount": 25000.00,
  "scores": {
    "overall_fraud_score": 0.9412,
    "syndicate_score": 0.7840,
    "anomaly_transaction_score": 0.8921,
    "velocity_score": 0.8650
  },
  "location": {
    "payer_to_merchant_km": 621.4,
    "terminal_speed_kmh": 0.0,
    "is_terminal_cloned": false
  },
  "seasonal_context": {
    "mcc_surge_multiplier": 1.0,
    "unique_payer_ratio": 0.25
  },
  "action": "BLOCK",
  "shap_explanations": [
    "[Syndicate Pillar] Score: 0.7840 | SHAP Impact: +1.4210",
    "[Anomaly Pillar]   Score: 0.8921 (MSE: 14.82) | SHAP Impact: +1.1045",
    "[Velocity Pillar]  Score: 0.8650 (8 tx/hr, Unique Payers: 25%, MCC Surge: 1.0x) | SHAP Impact: +0.9320",
    "[Geo Telemetry]    Payer-Merchant Dist: 621.4 km | Terminal Speed: 0.0 km/h"
  ]
}


Directory Structure

.
├── fraud_engine.py                     # Core modular ML, GNN, Geo, Seasonal, SHAP & Live Listener engine
├── live_website_generator.py           # Live local website checkout transaction generator
├── merchant_gnn_embeddings_master.npy  # Persistent 64-dimensional GraphSAGE merchant embeddings
├── transformer_temporal_model.pth      # Serialized PyTorch Transformer Autoencoder weights
├── catboost_master_fraud_model.cbm     # Saved CatBoost decision tree model file
├── live_incoming_tx.jsonl              # Live incoming transaction stream queue
├── transactions_ledger.jsonl           # Persistent append-only transaction & verdict audit log
└── project_readme.md                   # Documentation file


Local Execution and Integration Instructions

Prerequisites

Python 3.9 or higher installed.

Pre-trained model artifacts (merchant_gnn_embeddings_master.npy, transformer_temporal_model.pth, catboost_master_fraud_model.cbm) placed in the root directory.

Step 1: Install Dependencies

pip install numpy torch catboost


Step 2: Run Live Two-Terminal Demo

Terminal 1 (Start the Fraud Detection Engine Listener):

python fraud_engine.py


Terminal 2 (Start the Live Local Website Checkout Streamer):

python live_website_generator.py


Step 3: Integrate with Your Website

Direct Python Backend Integration:

from fraud_engine import FraudEngine

engine = FraudEngine()
verdict = engine.evaluate(incoming_transaction_dict)


HTTP Endpoint Testing with Generator:
If you expose FraudEngine.evaluate() behind a local website endpoint, point the generator directly to your URL:

python live_website_generator.py --http http://localhost:8000/api/v1/analyze-fraud


Cross-Language File Stream Integration (Node.js / PHP / Java):
Append completed payment JSON objects as single lines to live_incoming_tx.jsonl while python fraud_engine.py runs in the background.