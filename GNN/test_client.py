import os
import sys
import json
import time
import uuid
import random
from datetime import datetime, timezone

# Supports 3 modes seamlessly:
#   1. Default (File Stream): Appends live JSON events to 'live_incoming_tx.jsonl'
#   2. Direct Engine Mode (--direct): Imports FraudEngine and evaluates in-process
#   3. HTTP Webhook Mode (--http <url>): POSTs JSON payload to your website's API endpoint
STREAM_FILE = "live_incoming_tx.jsonl"

MERCHANTS = [
    {"id": "M_KURNOOL_CAFE_01", "mcc": 5812, "hw": "SOUNDBOX_V2_4G", "tid": "TID_KRN_101", "lat": 15.8281, "lon": 78.0373},
    {"id": "M_HYD_ELECTRONICS_02", "mcc": 5732, "hw": "POS_TERMINAL", "tid": "TID_HYD_202", "lat": 17.3850, "lon": 78.4867},
    {"id": "M_BLR_APPAREL_03", "mcc": 5651, "hw": "SOUNDBOX_V2_4G", "tid": "TID_BLR_303", "lat": 12.9716, "lon": 77.5946},
    {"id": "M_VJA_GROCERY_04", "mcc": 5411, "hw": "SOUNDBOX_V2_4G", "tid": "TID_VJA_404", "lat": 16.5062, "lon": 80.6480},
    {"id": "M_SPORTS_HUB_99", "mcc": 5941, "hw": "WEB_REST_API", "tid": "TID_WEB_999", "lat": 17.4401, "lon": 78.3489},
]


def generate_normal_payment() -> dict:
    m = random.choice(MERCHANTS[:4])
    payer_lat = m["lat"] + random.uniform(-0.02, 0.02)
    payer_lon = m["lon"] + random.uniform(-0.02, 0.02)
    return {
        "raw_event_id": f"PAY_{uuid.uuid4().hex[:8].upper()}",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "merchant_id": m["id"],
        "payer_id": f"UPI_USER_{random.randint(10000, 99999)}",
        "raw_amount": float(random.randint(12000, 185000)),
        "mcc_raw": m["mcc"],
        "terminal_hardware_type": m["hw"],
        "terminal_id": m["tid"],
        "merchant_lat": m["lat"],
        "merchant_lon": m["lon"],
        "payer_lat": round(payer_lat, 5),
        "payer_lon": round(payer_lon, 5),
    }


def generate_cloned_pos_attack() -> dict:
    m = MERCHANTS[0]  # Physical terminal TID_KRN_101 suddenly appears in New Delhi
    return {
        "raw_event_id": f"PAY_{uuid.uuid4().hex[:8].upper()}",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "merchant_id": m["id"],
        "payer_id": f"UPI_USER_{random.randint(10000, 99999)}",
        "raw_amount": 450000.0,
        "mcc_raw": m["mcc"],
        "terminal_hardware_type": m["hw"],
        "terminal_id": m["tid"],
        "merchant_lat": 28.6139,
        "merchant_lon": 77.2090,
        "payer_lat": 28.6150,
        "payer_lon": 77.2100,
    }


def generate_syndicate_burst() -> list:
    m = MERCHANTS[4]  # Isolated sports merchant M_SPORTS_HUB_99
    mules = ["MULE_RING_ALPHA", "MULE_RING_BETA"]
    burst = []
    for _ in range(8):
        burst.append({
            "raw_event_id": f"PAY_{uuid.uuid4().hex[:8].upper()}",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "merchant_id": m["id"],
            "payer_id": random.choice(mules),
            "raw_amount": 2500000.0,
            "mcc_raw": m["mcc"],
            "terminal_hardware_type": "WEB_REST_API",
            "terminal_id": m["tid"],
            "merchant_lat": m["lat"],
            "merchant_lon": m["lon"],
            "payer_lat": 19.0760,
            "payer_lon": 72.8777,
        })
    return burst


def run_generator():
    direct_mode = "--direct" in sys.argv
    http_url = None
    if "--http" in sys.argv:
        idx = sys.argv.index("--http")
        http_url = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "http://localhost:8000/api/v1/analyze-fraud"

    engine = None
    if direct_mode:
        from fraud_engine import FraudEngine
        engine = FraudEngine()
        print("[WEBSITE GENERATOR] Running in DIRECT ENGINE mode.\n")
    elif http_url:
        import urllib.request
        print(f"[WEBSITE GENERATOR] Streaming live JSON payloads via HTTP POST to {http_url}\n")
    else:
        print(f"[WEBSITE GENERATOR] Streaming live checkout events to '{STREAM_FILE}'... (Ctrl+C to stop)\n")

    def dispatch(tx: dict, label: str):
        if direct_mode and engine:
            engine.evaluate(tx)
        elif http_url:
            req = urllib.request.Request(
                http_url,
                data=json.dumps(tx).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                print(f"[WEBSITE -> API] {tx['raw_event_id']} ({label}) -> HTTP {resp.status}: {resp.read().decode()}")
        else:
            with open(STREAM_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(tx) + "\n")
                f.flush()
            print(
                f"[WEBSITE STREAM] {tx['raw_event_id']} | {tx['merchant_id']} | "
                f"Amt: {tx['raw_amount']/100:.2f} | Pattern: {label}"
            )

    counter = 0
    try:
        while True:
            counter += 1
            if counter % 12 == 0:
                for fraud_tx in generate_syndicate_burst():
                    dispatch(fraud_tx, "ISOLATED_SYNDICATE_BURST")
                    time.sleep(0.25)
            elif counter % 7 == 0:
                dispatch(generate_cloned_pos_attack(), "CLONED_POS_IMPOSSIBLE_TRAVEL")
            else:
                dispatch(generate_normal_payment(), "NORMAL_CHECKOUT")

            time.sleep(random.uniform(0.8, 2.5))
    except KeyboardInterrupt:
        if engine:
            engine.graph.save()
        print("\n[WEBSITE GENERATOR] Stopped.")


if __name__ == "__main__":
    run_generator()