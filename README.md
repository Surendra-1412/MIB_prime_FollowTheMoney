# PayU Protect &middot; MIB Analyst Workspace
### Real-Time Explainable Multi-Pillar Machine Learning & OSINT Fraud Detection Pipeline

## Architecture Overview

```
+-----------------------------------------------------------------------------------------+
|                                LIVE INCOMING TRANSACTION STREAM                         |
|             { event_id, merchant_id, payer_id, amount, mcc, geo_coords, tid }           |
+--------------------------------------------+--------------------------------------------+
                                             |
            +--------------------------------+--------------------------------+
            |                                |                                |
            v                                v                                v
+-----------------------+        +-----------------------+        +-----------------------+
|       PILLAR 1        |        |       PILLAR 2        |        |       PILLAR 3        |
|     Spatial GNN       |        | Temporal Transformer  |        |  Contextual & Geo     |
| (Dynamic GraphSAGE)   |        |  (10-Step Autoencoder)|        | (Haversine & Seasonal)|
+-----------+-----------+        +-----------+-----------+        +-----------+-----------+
            |                                |                                |
     64D Node Embedding &              Sequence MSE &                  Adaptive Velocity &
     Syndicate Score                   Anomaly Score                   Impossible Travel (>1000km/h)
            |                                |                                |
            +--------------------------------+--------------------------------+
                                             |
                                             v
                        +-----------------------------------------+
                        |      CatBoost Classifier (Fusion)       |
                        |      catboost_master_fraud_model.cbm    |
                        +--------------------+--------------------+
                                             |
                                             v
                        +-----------------------------------------+
                        |       SHAP Explainability Module        |
                        |       Pillar Attribution Vectors        |
                        +--------------------+--------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                         MIB ANALYST WORKSPACE & PERSISTENT LEDGER                       |
|           { overall_fraud_score, syndicate_score, anomaly_score, velocity_score }       |
+-----------------------------------------------------------------------------------------+
                                             ^
                                             |
             +-------------------------------+-------------------------------+
             |                                                               |
+----------------------------+                                 +----------------------------+
|          PILLAR 4          |                                 |          PILLAR 5          |
|    Document Extraction     |                                 |  Public Information OSINT  |
|      (PyMuPDF NLP)         |                                 |   (RDAP & Domain Lookup)   |
| GSTIN, PAN, IFSC, Invoices |                                 | Domain Age, Registrar, DNS |
+----------------------------+                                 +----------------------------+
```

---

## 1. Multi-Pillar Intelligence Engines

1. **Spatial Intelligence (Dynamic GraphSAGE GNN)**:
   - Evaluates 64-dimensional GraphSAGE merchant embeddings (`merchant_gnn_embeddings_master.npy`).
   - Dynamically adds unseen merchants, propagates risk across shared payers via inductive neighbor aggregation, and reinforces risk subspaces.
   - Generates the normalized Syndicate Risk Score ($S_{syn} \in [0, 1]$).

2. **Temporal Intelligence (PyTorch Transformer Autoencoder)**:
   - Evaluates trailing 10-step transaction cadence and amount sequences (`transformer_temporal_model.pth`).
   - Normalizes sequence reconstruction Mean Squared Error (MSE) into the Anomaly Transaction Score ($S_{anom} \in [0, 1]$).

3. **Contextual, Seasonal & Geospatial Telemetry (Haversine Tracker)**:
   - Computes Haversine distance between merchant coordinates and payer coordinates.
   - Flags cloned POS terminals if velocity implies travel speeds $> 1,000\text{ km/h}$.
   - Self-calibrating Merchant Category Code (MCC) macro-surge tracker distinguishes platform-wide festive rushes from isolated spikes.

4. **Document Consistency & Entity Extraction (NLP Model)**:
   - Extracts structured regulatory and banking entities from invoices, GST registrations, and bank statements (`nlp_model.py` via PyMuPDF).
   - Validates GSTIN format, PAN matching, IFSC codes, bank account numbers, invoice references, and financial totals.

5. **Public Information Verification (OSINT RDAP Domain Intelligence)**:
   - Extracts domains from evidence documents, emails, and merchant profiles (`osint_analyzer.py`).
   - Queries authentic Registration Data Access Protocol (RDAP) endpoints and DNS resolvers for creation date, domain age in days, registrar, privacy shields, and MX mail capability.
   - Flags newly registered domains ($< 90$ days old), anonymous proxy shields, and suspended domains.

6. **CatBoost Fusion & Explainability**:
   - Concatenates 73 features (64 GNN dimensions + 9 contextual/temporal attributes) into `catboost_master_fraud_model.cbm`.
   - Computes overall fraud probability and decomposes decisions into human-readable SHAP pillar attributions.

---

## 2. Real Model-Derived Queries

The generated queries in the MIB Analyst Workspace are dynamically produced from the multi-pillar engine results:
- **Query 01**: `Public Information Verification (RDAP Domain OSINT)` &mdash; Domain age, registrar, IP resolution, privacy proxy flags.
- **Query 02**: `Transaction Behaviour Analysis (Transformer Autoencoder)` &mdash; 10-step cadence MSE reconstruction error and anomaly score.
- **Query 03**: `Document Consistency & Extraction (NLP Extractor)` &mdash; Extracted GSTIN, PAN, IFSC, and invoice totals.
- **Query 04**: `Beneficiary Network Analysis (GraphSAGE GNN)` &mdash; 64D node embedding norm, syndicate score, neighbor clustering.
- **Query 05**: `Historical Risk & Geospatial Telemetry (CatBoost Fusion)` &mdash; Overall fraud probability, Haversine impossible travel speed, and MCC surge multiplier.

---

## 3. Starting the Integrated System

### Run the Server
```bash
python server.py 8080
```
The server will bind to `http://localhost:8080`, initialize all models (`catboost_master_fraud_model.cbm`, `merchant_gnn_embeddings_master.npy`, `transformer_temporal_model.pth`), and serve both the REST API and the analyst frontend.

### Open the Workspace
Open your web browser and navigate to:
```
http://localhost:8080
```

### Live Test Scenarios
In the top simulation bar, test one-click fraud scenarios:
- **Normal Payment**: Tests normal checkout cadence (CatBoost outputs low fraud probability $\sim 6.8\%$, `ALLOW`).
- **Cloned POS Terminal**: Tests impossible travel $> 1,000\text{ km/h}$ from Chennai to New Delhi (outputs $99\%$ fraud probability, `BLOCK`).
- **Syndicate Burst**: Tests high-value burst linked to `MULE_RING_ALPHA` (GNN updates risk subspace, CatBoost flags syndicate anomaly, `BLOCK`).

### Evidence Document & Domain OSINT
Click the **Evidence Document & OSINT** button to view or input custom invoices, receipts, or contact information. Click **Run Document NLP & RDAP OSINT** to extract all tax and banking entities and perform instant RDAP domain queries.
