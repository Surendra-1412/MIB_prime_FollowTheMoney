import os
import sys
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

# Ensure PAY_U_HACK and PayUprotect are in Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HACK_DIR = os.path.join(CURRENT_DIR, "PAY_U_HACK")
for p in [CURRENT_DIR, HACK_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fraud_engine import FraudEngine
from nlp_model import MerchantDocumentExtractor
from osint_analyzer import OSINTDomainAnalyzer

# ==============================================================================
# 1. CORE DATA STORE & STATE
# ==============================================================================
GNN_PATH = os.path.join(HACK_DIR, "merchant_gnn_embeddings_master.npy")
TRANSFORMER_PATH = os.path.join(HACK_DIR, "transformer_temporal_model.pth")
CATBOOST_PATH = os.path.join(HACK_DIR, "catboost_master_fraud_model.cbm")
LEDGER_PATH = os.path.join(HACK_DIR, "transactions_ledger.jsonl")

# Initialize models
fraud_engine = FraudEngine(
    gnn_path=GNN_PATH,
    transformer_path=TRANSFORMER_PATH,
    catboost_path=CATBOOST_PATH,
    ledger_path=LEDGER_PATH,
    print_terminal=True,
)
nlp_extractor = MerchantDocumentExtractor()
osint_analyzer = OSINTDomainAnalyzer()

# Seed merchants
MERCHANTS_DB: Dict[str, Dict[str, Any]] = {
    "MIB-001": {
        "id": "MIB-001",
        "name": "Merchant A (Tech Retail)",
        "category": "Electronics Retail",
        "city": "Chennai, Tamil Nadu",
        "registration": "CIN U52100TN2019PTC128841, 2019",
        "verification": "verified",
        "mri": 890,
        "tri": 860,
        "domain": "merchanta-tech.in",
        "mcc": 5732,
        "hardware": "POS_TERMINAL",
        "tid": "TID_CHN_101",
        "lat": 13.0827,
        "lon": 80.2707,
        "document_text": (
            "TAX INVOICE\n"
            "M/s Merchant A Technologies Private Limited\n"
            "GSTIN: 33AABCM1234F1Z5\n"
            "PAN: AABCM1234F\n"
            "Bank Account: 50200012345678\n"
            "IFSC: HDFC0001234\n"
            "Invoice No: INV-2026-8812\n"
            "Date: 12-Nov-2026\n"
            "Contact: support@merchanta-tech.in | +91 9840123456\n"
            "Total Amount: INR 45,000.00\n"
            "CGST @ 9%: INR 4,050.00\n"
            "SGST @ 9%: INR 4,050.00\n"
            "Grand Total: INR 53,100.00\n"
        ),
    },
    "MIB-002": {
        "id": "MIB-002",
        "name": "Merchant B (Travel Services)",
        "category": "Travel Services",
        "city": "Bengaluru, Karnataka",
        "registration": "CIN U63090KA2016PTC094512, 2016",
        "verification": "verified",
        "mri": 815,
        "tri": 790,
        "domain": "travelhub-bengaluru.com",
        "mcc": 4722,
        "hardware": "WEB_REST_API",
        "tid": "TID_BLR_202",
        "lat": 12.9716,
        "lon": 77.5946,
        "document_text": (
            "STATEMENT OF ACCOUNT / GST REGISTRATION\n"
            "Legal Name: Travel Services Karnataka Private Limited\n"
            "GSTIN: 29AABCT5678G1Z2\n"
            "PAN: AABCT5678G\n"
            "Account No: 919010098765432\n"
            "IFSC: SBIN0004567\n"
            "Contact: bookings@travelhub-bengaluru.com | +91 9880123456\n"
            "Registration Date: 20-Aug-2016\n"
            "Net Settlement Amount: INR 128,450.00\n"
        ),
    },
    "MIB-003": {
        "id": "MIB-003",
        "name": "Merchant C (Rapid Digital Goods)",
        "category": "Digital Goods",
        "city": "Mumbai, Maharashtra",
        "registration": "CIN U72900MH2024PTC412207, 2024",
        "verification": "flagged",
        "mri": 245,
        "tri": 180,
        "domain": "quickpay-digitalmall.com",
        "mcc": 5815,
        "hardware": "SOUNDBOX_V2_4G",
        "tid": "TID_MUM_303",
        "lat": 19.0760,
        "lon": 72.8777,
        "document_text": (
            "ESTIMATE BILL / TEMPORARY SLIP\n"
            "Business Name: QuickPay Digital Mall\n"
            "GSTIN: 27ABCDE9999Z1Z9\n"
            "PAN: ABCDE9999Z\n"
            "Acc: 110022334455\n"
            "IFSC: UTIB0000999\n"
            "Inv ID: REF-9901\n"
            "Email: payments@quickpay-digitalmall.com\n"
            "Total: INR 250,000.00\n"
        ),
    },
    "M_SPORTS_HUB_99": {
        "id": "M_SPORTS_HUB_99",
        "name": "Sports Hub Syndicate Ring",
        "category": "Sporting Goods",
        "city": "Hyderabad, Telangana",
        "registration": "Unregistered / Offshore Proxy",
        "verification": "flagged",
        "mri": 190,
        "tri": 140,
        "domain": "sportshub99.com",
        "mcc": 5941,
        "hardware": "WEB_REST_API",
        "tid": "TID_WEB_999",
        "lat": 17.4401,
        "lon": 78.3489,
        "document_text": (
            "MEMO OF SALE\n"
            "Trade Name: Sports Hub 99\n"
            "PAN: ZZZZZ9999Z\n"
            "Email: admin@sportshub99.com\n"
            "Amount: INR 2,500,000.00\n"
        ),
    },
}

QUERIES_STORE: Dict[str, List[Dict[str, Any]]] = {}
AUDIT_STORE: Dict[str, List[Dict[str, Any]]] = {
    "MIB-001": [
        {"time": "2026-11-12T14:10:00Z", "text": "Engine verified GNN embedding and 2019 RDAP registration."},
        {"time": "2026-11-10T09:30:00Z", "text": "Merchant onboarded as MIB-001 with verified GSTIN 33AABCM1234F1Z5."},
    ],
    "MIB-002": [
        {"time": "2026-11-11T16:00:00Z", "text": "Transformer cadence normal (MSE: 0.12). Low anomaly risk."},
    ],
    "MIB-003": [
        {"time": "2026-11-12T18:05:00Z", "text": "CRITICAL: RDAP detected domain registered 40 days ago under clientHold."},
        {"time": "2026-11-12T18:02:00Z", "text": "CatBoost flagged high fraud score (0.9412) due to rapid velocity and MSE anomaly."},
    ],
    "M_SPORTS_HUB_99": [
        {"time": "2026-11-12T18:23:00Z", "text": "CRITICAL: Bipartite GNN flagged mule syndicate linkage (MULE_RING_ALPHA)."},
    ],
}


def add_audit(mid: str, text: str):
    now_iso = datetime.now(timezone.utc).isoformat()
    if mid not in AUDIT_STORE:
        AUDIT_STORE[mid] = []
    AUDIT_STORE[mid].insert(0, {"time": now_iso, "text": text})


# ==============================================================================
# 2. ENGINE EVALUATION & DYNAMIC QUERY SYNTHESIS
# ==============================================================================
def execute_engine_for_merchant(merchant_id: str, custom_tx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Executes the full multi-pillar fraud detection pipeline:
    1. Spatial GNN Embedding & Syndicate Graph Update
    2. PyTorch Temporal Transformer Autoencoder Sequence MSE Anomaly
    3. Contextual, Haversine impossible travel, and MCC seasonal surge tracker
    4. CatBoost Master Classifier prediction with SHAP impact attribution
    5. NLP Document Extractor KYC/KYB entity verification
    6. OSINT RDAP / DNS domain lookup
    Generates authentic, model-derived queries and updates MRI / TRI scores.
    """
    m = MERCHANTS_DB.get(merchant_id)
    if not m:
        raise ValueError(f"Merchant '{merchant_id}' not found.")

    # 1. Prepare sample or live transaction
    if not custom_tx:
        if merchant_id == "MIB-001":
            tx = {
                "raw_event_id": f"PAY_M1_{uuid.uuid4().hex[:6].upper()}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "merchant_id": merchant_id,
                "payer_id": f"UPI_USER_{int(time.time()) % 10000}",
                "raw_amount": 53100.0,
                "mcc_raw": m.get("mcc", 5732),
                "terminal_hardware_type": m.get("hardware", "POS_TERMINAL"),
                "terminal_id": m.get("tid", "TID_CHN_101"),
                "merchant_lat": m.get("lat", 13.0827),
                "merchant_lon": m.get("lon", 80.2707),
                "payer_lat": 13.0830,
                "payer_lon": 80.2710,
            }
        elif merchant_id == "MIB-002":
            tx = {
                "raw_event_id": f"PAY_M2_{uuid.uuid4().hex[:6].upper()}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "merchant_id": merchant_id,
                "payer_id": f"UPI_USER_{int(time.time()) % 10000}",
                "raw_amount": 128450.0,
                "mcc_raw": m.get("mcc", 4722),
                "terminal_hardware_type": m.get("hardware", "WEB_REST_API"),
                "terminal_id": m.get("tid", "TID_BLR_202"),
                "merchant_lat": m.get("lat", 12.9716),
                "merchant_lon": m.get("lon", 77.5946),
                "payer_lat": 12.9720,
                "payer_lon": 77.5950,
            }
        elif merchant_id == "MIB-003":
            tx = {
                "raw_event_id": f"PAY_M3_{uuid.uuid4().hex[:6].upper()}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "merchant_id": merchant_id,
                "payer_id": "MULE_RING_BETA",
                "raw_amount": 250000.0,
                "mcc_raw": m.get("mcc", 5815),
                "terminal_hardware_type": m.get("hardware", "SOUNDBOX_V2_4G"),
                "terminal_id": m.get("tid", "TID_MUM_303"),
                "merchant_lat": m.get("lat", 19.0760),
                "merchant_lon": m.get("lon", 72.8777),
                "payer_lat": 28.6139,  # Impossible travel distance (> 1000 km)
                "payer_lon": 77.2090,
            }
        else:  # M_SPORTS_HUB_99
            tx = {
                "raw_event_id": f"PAY_HUB_{uuid.uuid4().hex[:6].upper()}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "merchant_id": merchant_id,
                "payer_id": "MULE_RING_ALPHA",
                "raw_amount": 2500000.0,
                "mcc_raw": m.get("mcc", 5941),
                "terminal_hardware_type": m.get("hardware", "WEB_REST_API"),
                "terminal_id": m.get("tid", "TID_WEB_999"),
                "merchant_lat": m.get("lat", 17.4401),
                "merchant_lon": m.get("lon", 78.3489),
                "payer_lat": 19.0760,
                "payer_lon": 72.8777,
            }
    else:
        tx = custom_tx

    # 2. Run Fraud Detection Engine (GNN + Transformer + CatBoost + SHAP)
    engine_verdict = fraud_engine.evaluate(tx)
    scores = engine_verdict["scores"]
    overall_fraud_score = scores["overall_fraud_score"]
    syndicate_score = scores["syndicate_score"]
    anomaly_score = scores["anomaly_transaction_score"]
    velocity_score = scores["velocity_score"]
    shap_exps = engine_verdict.get("shap_explanations", [])
    action = engine_verdict.get("action", "REVIEW")
    loc = engine_verdict.get("location", {})

    # 3. Run NLP Document Extraction
    doc_text = m.get("document_text", "")
    nlp_results = nlp_extractor.extract_features(doc_text, is_raw_text=True, file_name=f"{merchant_id}_evidence.txt")
    doc_profile = nlp_results.get("business_profile", {})
    banking = nlp_results.get("banking_details", {})
    financial = nlp_results.get("financial_and_invoice_details", {})

    # 4. Run OSINT RDAP / Domain Lookup
    osint_results = osint_analyzer.analyze_document_or_merchant(
        doc_features=nlp_results, raw_text=doc_text, merchant_meta=m
    )

    # 5. Calculate calibrated MRI (Merchant Risk Index) and TRI (Transaction Risk Index)
    # Higher is healthier (0-1000 scale)
    calc_mri = int(round(1000 * (1.0 - (0.65 * overall_fraud_score + 0.35 * osint_results["risk_score"]))))
    calc_tri = int(round(1000 * (1.0 - (0.50 * anomaly_score + 0.30 * velocity_score + 0.20 * (1.0 if loc.get("is_terminal_cloned") else 0.0)))))
    calc_mri = max(50, min(980, calc_mri))
    calc_tri = max(50, min(980, calc_tri))
    m["mri"] = calc_mri
    m["tri"] = calc_tri
    m["verification"] = "flagged" if (action == "BLOCK" or calc_mri < 400) else ("review" if (action == "REVIEW" or calc_mri < 700) else "verified")

    # 6. Synthesize Real Queries from the Engine Pillars
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Query 1: OSINT Domain Intelligence
    osint_risk = osint_results["risk_level"]
    osint_status = "Review" if osint_risk in ("High", "Critical") else "Verified"
    q1 = {
        "id": f"{merchant_id}-Q1",
        "no": 1,
        "title": "Public Information Verification (RDAP Domain OSINT)",
        "category": "OSINT",
        "tags": ["osint"],
        "weight": 15,
        "evidence": osint_results["evidence_count"],
        "status": osint_status,
        "risk": osint_risk,
        "generated": today_str,
        "detail": {
            "evidence": (
                f"Domain: {osint_results['domain']} | Registrar: {osint_results['registrar']} | "
                f"Age: {osint_results['domain_age_days']} days | IP: {osint_results['ip_address']} | "
                f"Privacy Shield: {'Enabled (Masked)' if osint_results['privacy_shield'] else 'Disabled (Transparent)'}"
            ),
            "contribution": f"{round(15 * (1000 - calc_mri) / 100)} points MRI deduction (15% weight)",
            "source": f"RDAP Protocol (IANA / {osint_results['registrar']})",
            "observations": osint_results["observations"],
        },
    }

    # Query 2: Transformer Temporal Anomaly
    trans_risk = "Critical" if anomaly_score > 0.70 else ("High" if anomaly_score > 0.45 else ("Medium" if anomaly_score > 0.20 else "Low"))
    trans_status = "Review" if trans_risk in ("High", "Critical") else "Verified"
    q2_shap = next((s for s in shap_exps if "Anomaly Pillar" in s), f"Anomaly Score: {anomaly_score}")
    q2 = {
        "id": f"{merchant_id}-Q2",
        "no": 2,
        "title": "Transaction Behaviour Analysis (Transformer Autoencoder)",
        "category": "Transaction",
        "tags": ["txn", "ai"],
        "weight": 30,
        "evidence": 10,
        "status": trans_status,
        "risk": trans_risk,
        "generated": today_str,
        "detail": {
            "evidence": f"10 rolling transaction sequence steps evaluated via PyTorch Transformer Autoencoder. Normalized Anomaly Score: {anomaly_score:.4f}.",
            "contribution": f"{round(30 * (1000 - calc_mri) / 100)} points MRI deduction (30% weight)",
            "source": "Transformer Autoencoder (transformer_temporal_model.pth)",
            "observations": f"{q2_shap}. Rolling amounts cadence evaluated. Seasonal surge multiplier: {engine_verdict.get('seasonal_context', {}).get('mcc_surge_multiplier', 1.0)}x.",
        },
    }

    # Query 3: NLP Document Consistency
    doc_type = nlp_results.get("document_type", "GENERAL_EVIDENCE_DOCUMENT")
    gstin_cnt = len(doc_profile.get("gstin_list", []))
    pan_cnt = len(doc_profile.get("pan_list", []))
    ifsc_cnt = len(banking.get("ifsc_codes", []))
    amounts_cnt = len(financial.get("all_extracted_amounts", []))
    ev_doc_cnt = gstin_cnt + pan_cnt + ifsc_cnt + amounts_cnt + len(financial.get("invoice_or_ref_numbers", []))
    doc_consistent = bool(gstin_cnt > 0 and pan_cnt > 0 and amounts_cnt > 0)
    doc_risk = "Low" if doc_consistent else ("Medium" if (gstin_cnt or pan_cnt) else "High")
    doc_status = "Verified" if doc_consistent else "Pending"
    q3 = {
        "id": f"{merchant_id}-Q3",
        "no": 3,
        "title": "Document Consistency & Extraction (NLP Extractor)",
        "category": "Documents",
        "tags": ["doc", "ai"],
        "weight": 20,
        "evidence": max(ev_doc_cnt, 4),
        "status": doc_status,
        "risk": doc_risk,
        "generated": today_str,
        "detail": {
            "evidence": (
                f"Classified Doc: {doc_type} | Extracted GSTIN: {doc_profile.get('gstin_list', ['None'])} | "
                f"PAN: {doc_profile.get('pan_list', ['None'])} | IFSC: {banking.get('ifsc_codes', ['None'])} | "
                f"Invoice Primary Amount: INR {financial.get('primary_total_amount', 'N/A')}"
            ),
            "contribution": f"{round(20 * (1000 - calc_mri) / 100)} points MRI deduction (20% weight)",
            "source": "PyMuPDF / Regex Entity Engine (nlp_model.py)",
            "observations": (
                f"Document verified as '{doc_type}'. Business legal entity: {doc_profile.get('business_name', m['name'])}. "
                f"PAN/GSTIN correlation {'matches registered state format' if doc_consistent else 'incomplete KYC tax entity linkage'}."
            ),
        },
    }

    # Query 4: Dynamic GNN Syndicate Analysis
    gnn_risk = "Critical" if syndicate_score > 0.65 else ("High" if syndicate_score > 0.40 else ("Medium" if syndicate_score > 0.20 else "Low"))
    gnn_status = "Review" if gnn_risk in ("High", "Critical") else "Verified"
    q4_shap = next((s for s in shap_exps if "Syndicate Pillar" in s), f"Syndicate Score: {syndicate_score}")
    q4 = {
        "id": f"{merchant_id}-Q4",
        "no": 4,
        "title": "Beneficiary Network Analysis (GraphSAGE GNN)",
        "category": "Beneficiary",
        "tags": ["benef", "network"],
        "weight": 20,
        "evidence": 64,  # 64 GNN embedding features
        "status": gnn_status,
        "risk": gnn_risk,
        "generated": today_str,
        "detail": {
            "evidence": f"64-dimensional GraphSAGE merchant embedding evaluated. Syndicate risk score: {syndicate_score:.4f}. Is new graph node: {engine_verdict.get('is_new_merchant_node')}.",
            "contribution": f"{round(20 * (1000 - calc_mri) / 100)} points MRI deduction (20% weight)",
            "source": "Dynamic Merchant Graph (merchant_gnn_embeddings_master.npy)",
            "observations": f"{q4_shap}. Bipartite graph neighbor aggregation through shared payer '{tx.get('payer_id')}'. Risk subspace reinforcement applied.",
        },
    }

    # Query 5: CatBoost Fusion & Geospatial Telemetry
    geo_risk = "Critical" if loc.get("is_terminal_cloned") else ("High" if overall_fraud_score > 0.70 else ("Medium" if overall_fraud_score > 0.40 else "Low"))
    geo_status = "Review" if geo_risk in ("High", "Critical") else "Verified"
    q5_shap = next((s for s in shap_exps if "Velocity Pillar" in s), f"Velocity Score: {velocity_score}")
    q5 = {
        "id": f"{merchant_id}-Q5",
        "no": 5,
        "title": "Historical Risk & Geospatial Telemetry (CatBoost Fusion)",
        "category": "Historical Risk",
        "tags": ["hist"],
        "weight": 15,
        "evidence": 9,
        "status": geo_status,
        "risk": geo_risk,
        "generated": today_str,
        "detail": {
            "evidence": (
                f"Payer-to-Merchant: {loc.get('payer_to_merchant_km', 0.0)} km | Terminal Speed: {loc.get('terminal_speed_kmh', 0.0)} km/h | "
                f"Terminal Cloned: {loc.get('is_terminal_cloned', False)} | CatBoost Master Probability: {overall_fraud_score:.4f}"
            ),
            "contribution": f"{round(15 * (1000 - calc_mri) / 100)} points MRI deduction (15% weight)",
            "source": "CatBoost Classifier (catboost_master_fraud_model.cbm)",
            "observations": f"{q5_shap}. Action verdict: {action}. Overall model probability: {overall_fraud_score * 100:.1f}%.",
        },
    }

    queries_list = [q1, q2, q3, q4, q5]
    QUERIES_STORE[merchant_id] = queries_list

    # Record Audit log
    audit_msg = (
        f"Multi-Pillar Engine evaluated event {tx.get('raw_event_id')} -> Verdict: {action} "
        f"(Fraud: {overall_fraud_score:.4f}, Syndicate: {syndicate_score:.4f}, Anomaly: {anomaly_score:.4f}, "
        f"RDAP Age: {osint_results['domain_age_days']}d). MRI calibrated to {calc_mri}/1000."
    )
    add_audit(merchant_id, audit_msg)

    return {
        "merchant": m,
        "scores": scores,
        "verdict": engine_verdict,
        "osint": osint_results,
        "nlp": nlp_results,
        "queries": queries_list,
    }


# Ensure queries exist for initial merchants
for mid in list(MERCHANTS_DB.keys()):
    try:
        execute_engine_for_merchant(mid)
    except Exception as err:
        print(f"[WARN] Initializing {mid} failed: {err}")


# ==============================================================================
# 3. HTTP SERVER & API ROUTER
# ==============================================================================
class MIBRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=CURRENT_DIR, **kwargs)

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # API Routes
        if path == "/api/merchants" or path == "/merchants":
            self._send_json(list(MERCHANTS_DB.values()))
            return

        if path.startswith("/api/merchants/") or path.startswith("/merchants/"):
            parts = path.strip("/").split("/")
            # e.g., ["api", "merchants", "MIB-001", "queries"] or ["merchants", "MIB-001", "queries"]
            if parts[0] == "api":
                parts.pop(0)
            if len(parts) >= 2:
                mid = parts[1]
                if len(parts) == 2:
                    m = MERCHANTS_DB.get(mid)
                    if m:
                        self._send_json(m)
                    else:
                        self._send_json({"error": "Merchant not found"}, status=404)
                    return
                elif len(parts) == 3:
                    sub = parts[2]
                    if sub == "queries":
                        qs = QUERIES_STORE.get(mid, [])
                        self._send_json(qs)
                        return
                    elif sub == "audit":
                        aud = AUDIT_STORE.get(mid, [])
                        self._send_json(aud)
                        return

        # Fallback to standard static file server
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else ""
        payload = json.loads(body) if body else {}

        # 1. POST /merchants/:id/engine/run or /api/merchants/:id/engine/run
        if "/engine/run" in path:
            parts = path.strip("/").split("/")
            if parts[0] == "api":
                parts.pop(0)
            mid = parts[1]
            try:
                res = execute_engine_for_merchant(mid)
                self._send_json({"ok": True, "data": res})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=500)
            return

        # 2. POST /queries/:id/decision or /api/queries/:id/decision
        if "/decision" in path:
            parts = path.strip("/").split("/")
            if parts[0] == "api":
                parts.pop(0)
            qid = parts[1]
            decision = payload.get("decision", "verify")
            comment = payload.get("comment", "")
            mid = qid.split("-Q")[0]

            qs = QUERIES_STORE.get(mid, [])
            target_q = next((q for q in qs if q["id"] == qid), None)
            if target_q:
                status_map = {"verify": "Verified", "reject": "Rejected", "qualify": "Qualified"}
                target_q["status"] = status_map.get(decision, "Verified")

            add_audit(
                mid,
                f"Analyst marked {qid} as {decision.capitalize()}" + (f': "{comment}"' if comment else ""),
            )
            self._send_json({"ok": True})
            return

        # 3. POST /api/merchants/:id/document (Document upload / put document analysis)
        if "/document" in path:
            parts = path.strip("/").split("/")
            if parts[0] == "api":
                parts.pop(0)
            mid = parts[1]
            doc_text = payload.get("text", "")
            m = MERCHANTS_DB.get(mid)
            if m and doc_text:
                m["document_text"] = doc_text
                # Run extraction and OSINT
                nlp_out = nlp_extractor.extract_features(doc_text, is_raw_text=True, file_name=f"{mid}_uploaded.txt")
                osint_out = osint_analyzer.analyze_document_or_merchant(doc_features=nlp_out, raw_text=doc_text, merchant_meta=m)
                execute_engine_for_merchant(mid)
                add_audit(mid, f"Document uploaded. NLP extracted entities, OSINT analyzed domain '{osint_out['domain']}'.")
                self._send_json({"ok": True, "nlp": nlp_out, "osint": osint_out})
            else:
                self._send_json({"ok": False, "error": "Invalid merchant or missing document text"}, status=400)
            return

        # 4. POST /api/analyze-transaction
        if path in ("/api/analyze-transaction", "/analyze-transaction"):
            try:
                mid = payload.get("merchant_id", "MIB-001")
                res = execute_engine_for_merchant(mid, custom_tx=payload)
                self._send_json({"ok": True, "verdict": res["verdict"], "scores": res["scores"]})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=500)
            return

        self._send_json({"error": "Endpoint not found"}, status=404)


def run_server(port: int = 8080):
    candidate_ports = [port, 8088, 5000, 3000, 8888]
    httpd = None
    active_port = port
    for p in candidate_ports:
        try:
            server_address = ("127.0.0.1", p)
            httpd = HTTPServer(server_address, MIBRequestHandler)
            active_port = p
            break
        except Exception as err:
            print(f"[MIB SERVER] Port {p} unavailable ({err}), trying next candidate...")

    if not httpd:
        raise RuntimeError("Could not bind to any candidate port.")

    print("=" * 80)
    print(f"[MIB BACKEND SERVER] Running successfully at http://localhost:{active_port}")
    print(f"API endpoints: /api/merchants, /api/merchants/<id>/queries, /api/merchants/<id>/engine/run")
    print(f"Serving UI frontend from: {CURRENT_DIR}")
    print("=" * 80)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[MIB BACKEND SERVER] Shutting down.")
        httpd.server_close()


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(p)
