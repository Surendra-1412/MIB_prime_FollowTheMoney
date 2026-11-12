import os
import sys
import json
import math
import time
import uuid
import zlib
from datetime import datetime, timezone
from collections import deque, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from catboost import CatBoostClassifier, Pool


# ==============================================================================
# 1. GEO-LOCATION & IMPOSSIBLE TRAVEL TRACKER
# ==============================================================================
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance in kilometers between two GPS coordinates."""
    if None in (lat1, lon1, lat2, lon2):
        return 0.0
    r = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class TerminalLocationTracker:
    def __init__(self):
        self.terminal_history: Dict[str, Tuple[float, float, float]] = {}

    def evaluate_location(
        self, terminal_id: str, m_lat: float, m_lon: float, p_lat: float, p_lon: float, current_ts: float
    ) -> dict:
        payer_dist_km = haversine_distance(p_lat, p_lon, m_lat, m_lon)
        travel_speed_kmh = 0.0
        is_cloned = False

        if terminal_id and m_lat is not None and m_lon is not None:
            if terminal_id in self.terminal_history:
                last_lat, last_lon, last_ts = self.terminal_history[terminal_id]
                dist_moved = haversine_distance(last_lat, last_lon, m_lat, m_lon)
                hours_elapsed = max((current_ts - last_ts) / 3600.0, 0.001)
                travel_speed_kmh = dist_moved / hours_elapsed
                is_cloned = travel_speed_kmh > 1000.0
            self.terminal_history[terminal_id] = (m_lat, m_lon, current_ts)

        return {
            "payer_to_merchant_km": round(payer_dist_km, 2),
            "terminal_speed_kmh": round(travel_speed_kmh, 2),
            "is_terminal_cloned": is_cloned,
        }


# ==============================================================================
# 2. DYNAMIC GNN GRAPH & ONLINE LEARNER
# ==============================================================================
class DynamicMerchantGraph:
    def __init__(self, gnn_path: str):
        self.gnn_path = gnn_path
        if os.path.exists(gnn_path):
            self.merchant_embeddings: Dict[str, np.ndarray] = np.load(gnn_path, allow_pickle=True).item()
        else:
            self.merchant_embeddings = {}

        self.payer_to_merchants: Dict[str, set] = defaultdict(set)
        self.merchant_to_payers: Dict[str, set] = defaultdict(set)

    def register_and_update_node(
        self, m_id: str, p_id: str, is_flagged_risky: bool = False, risk_signal: float = 0.0
    ) -> Tuple[np.ndarray, bool, float]:
        is_new_merchant = m_id not in self.merchant_embeddings
        if is_new_merchant:
            self.merchant_embeddings[m_id] = np.zeros(64, dtype=np.float32)

        self.payer_to_merchants[p_id].add(m_id)
        self.merchant_to_payers[m_id].add(p_id)

        # Inductive neighbor aggregation via shared payers
        neighbor_merchants = set()
        for linked_payer in self.merchant_to_payers[m_id]:
            neighbor_merchants.update(self.payer_to_merchants[linked_payer])
        neighbor_merchants.discard(m_id)

        current_emb = self.merchant_embeddings[m_id].copy()
        if neighbor_merchants:
            neighbor_vecs = [
                self.merchant_embeddings[nm]
                for nm in neighbor_merchants
                if nm in self.merchant_embeddings
            ]
            if neighbor_vecs:
                current_emb = 0.75 * current_emb + 0.25 * np.mean(neighbor_vecs, axis=0)

        # Online risk subspace reinforcement
        if is_flagged_risky or risk_signal > 0.65:
            rng = np.random.default_rng(zlib.crc32(m_id.encode("utf-8")))
            risk_direction = rng.uniform(0.05, 0.25, size=64).astype(np.float32)
            current_emb = 0.85 * current_emb + 0.15 * (risk_direction * risk_signal)

        self.merchant_embeddings[m_id] = current_emb

        emb_norm = float(np.linalg.norm(current_emb))
        raw_syndicate = 1.0 - math.exp(-0.35 * emb_norm)
        if len(neighbor_merchants) > 2:
            raw_syndicate = min(1.0, raw_syndicate + 0.10 * math.log1p(len(neighbor_merchants)))
        syndicate_score = round(float(np.clip(raw_syndicate, 0.0, 1.0)), 4)

        return current_emb, is_new_merchant, syndicate_score

    def save(self):
        np.save(self.gnn_path, self.merchant_embeddings)


# ==============================================================================
# 3. SELF-CALIBRATING SEASONAL & CATEGORY SURGE TRACKER
# ==============================================================================
class AdaptiveCategoryContextTracker:
    def __init__(self):
        self.mcc_1hr_events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50_000))
        self.mcc_baseline_ema: Dict[str, float] = defaultdict(lambda: 15.0)

    def update_and_get_surge(self, mcc: str, merchant_id: str, unix_ts: float) -> float:
        q = self.mcc_1hr_events[mcc]
        one_hour_ago = unix_ts - 3600.0
        while q and q[0][0] < one_hour_ago:
            q.popleft()
        q.append((unix_ts, merchant_id))

        total_mcc_tx = len(q)
        active_merchants = len({item[1] for item in q})
        avg_tx_per_merchant = total_mcc_tx / max(1, active_merchants)

        baseline = self.mcc_baseline_ema[mcc]
        if active_merchants >= 3:
            surge_ratio = avg_tx_per_merchant / max(1.0, baseline)
            self.mcc_baseline_ema[mcc] = 0.95 * baseline + 0.05 * avg_tx_per_merchant
        else:
            surge_ratio = 1.0

        return round(float(np.clip(surge_ratio, 1.0, 8.0)), 2)


# ==============================================================================
# 4. TEMPORAL TRANSFORMER AUTOENCODER
# ==============================================================================
class TransformerAutoencoder(nn.Module):
    def __init__(self, feature_dim=2, d_model=16, nhead=2, num_layers=2, seq_len=10):
        super().__init__()
        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=32, batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, feature_dim)

    def forward(self, x):
        out = self.embedding(x) + self.pos_encoder
        out = self.transformer(out)
        return self.decoder(out)


# ==============================================================================
# 5. CORE FRAUD DETECTION ENGINE (DIRECT CALL + LIVE FILE/STDIN STREAM)
# ==============================================================================
class FraudEngine:
    def __init__(
        self,
        gnn_path: str = "merchant_gnn_embeddings_master.npy",
        transformer_path: str = "transformer_temporal_model.pth",
        catboost_path: str = "catboost_master_fraud_model.cbm",
        ledger_path: str = "transactions_ledger.jsonl",
        print_terminal: bool = True,
    ):
        torch.set_num_threads(1)
        self.device = torch.device("cpu")
        self.ledger_path = ledger_path
        self.print_terminal = print_terminal

        self.graph = DynamicMerchantGraph(gnn_path=gnn_path)
        self.location_tracker = TerminalLocationTracker()
        self.category_tracker = AdaptiveCategoryContextTracker()

        self.transformer = TransformerAutoencoder().to(self.device)
        if os.path.exists(transformer_path):
            self.transformer.load_state_dict(
                torch.load(transformer_path, map_location=self.device, weights_only=True)
            )
        self.transformer.eval()

        self.cat_model = CatBoostClassifier()
        if os.path.exists(catboost_path):
            self.cat_model.load_model(catboost_path)

        self.rolling_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.hourly_activity: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))

        self.tabular_cols = [
            "amount",
            "is_soundbox",
            "mcc",
            "is_weekend",
            "is_round_amount",
            "hour_sin",
            "hour_cos",
            "merchant_1hr_velocity",
            "temporal_anomaly_score",
        ]
        self.all_feature_names = [f"gnn_{i}" for i in range(64)] + self.tabular_cols

    def _compute_temporal_scores(
        self, hist_queue: deque, current_dt: datetime, seasonal_scale: float
    ) -> Tuple[float, float]:
        if not hist_queue:
            return 0.0, 0.0

        amt_mean, amt_std = 50.0 * math.sqrt(seasonal_scale), 100.0 * math.sqrt(seasonal_scale)
        time_mean, time_std = 3600.0 / seasonal_scale, 7200.0 / seasonal_scale

        seq = np.zeros((10, 2), dtype=np.float32)
        prev_dt = current_dt
        items = list(reversed(hist_queue))
        start_idx = 10 - len(items)

        for idx, (h_amt, h_dt) in enumerate(items):
            delta = max((prev_dt - h_dt).total_seconds(), 0.1)
            seq[start_idx + idx, 0] = (h_amt - amt_mean) / amt_std
            seq[start_idx + idx, 1] = (delta - time_mean) / time_std
            prev_dt = h_dt

        tensor_x = torch.from_numpy(seq).unsqueeze(0)
        with torch.no_grad():
            reconstruction = self.transformer(tensor_x)
            raw_mse = float(torch.mean((reconstruction - tensor_x) ** 2).item())

        normalized_anomaly = 1.0 - math.exp(-0.15 * raw_mse)
        return round(raw_mse, 4), round(float(np.clip(normalized_anomaly, 0.0, 1.0)), 4)

    @staticmethod
    def _compute_velocity_score(
        velocity_1hr: int, unique_payer_ratio: float, category_surge_multiplier: float
    ) -> float:
        effective_velocity = velocity_1hr / max(1.0, category_surge_multiplier)
        diversity_penalty = 1.0 + (1.0 - unique_payer_ratio) * 2.5
        adjusted_velocity = effective_velocity * diversity_penalty

        if adjusted_velocity <= 8:
            return round(adjusted_velocity * 0.015, 4)

        score = 1.0 / (1.0 + math.exp(-0.08 * (adjusted_velocity - 45)))
        return round(float(np.clip(score, 0.0, 1.0)), 4)

    def _build_shap_explanations(
        self,
        feature_row: list,
        syndicate_score: float,
        anomaly_score: float,
        velocity_score: float,
        raw_mse: float,
        velocity_1hr: int,
        unique_payer_ratio: float,
        surge_mult: float,
        loc_info: dict,
        tx_features: dict,
    ) -> List[str]:
        pool = Pool(data=[feature_row], cat_features=[66])
        shap_values = self.cat_model.get_feature_importance(
            data=pool, type="ShapValues", thread_count=1
        )[0]
        contribs = shap_values[:-1]

        gnn_shap = float(np.sum(contribs[:64]))
        temp_shap = float(contribs[self.all_feature_names.index("temporal_anomaly_score")])
        vel_shap = float(contribs[self.all_feature_names.index("merchant_1hr_velocity")])

        explanations = [
            f"[Syndicate Pillar] Score: {syndicate_score:.4f} | SHAP Impact: {gnn_shap:+.4f}",
            f"[Anomaly Pillar]   Score: {anomaly_score:.4f} (MSE: {raw_mse:.2f}) | SHAP Impact: {temp_shap:+.4f}",
            f"[Velocity Pillar]  Score: {velocity_score:.4f} ({velocity_1hr} tx/hr, Unique Payers: {unique_payer_ratio:.0%}, "
            f"MCC Surge: {surge_mult}x) | SHAP Impact: {vel_shap:+.4f}",
            f"[Geo Telemetry]    Payer-Merchant Dist: {loc_info['payer_to_merchant_km']} km | "
            f"Terminal Speed: {loc_info['terminal_speed_kmh']} km/h",
        ]

        sorted_ctx = sorted(range(64, len(self.all_feature_names)), key=lambda i: abs(contribs[i]), reverse=True)
        for idx in sorted_ctx[:3]:
            fname = self.all_feature_names[idx]
            if fname in ("temporal_anomaly_score", "merchant_1hr_velocity"):
                continue
            val = contribs[idx]
            if abs(val) > 0.01:
                direction = "increases risk" if val > 0 else "reduces risk"
                explanations.append(
                    f"[Context Feature]  '{fname}' = {tx_features[fname]} | SHAP Impact: {val:+.4f} ({direction})"
                )
        return explanations

    def evaluate(self, raw_tx: dict) -> dict:
        """Evaluates a single transaction dictionary, logs it, and returns the verdict."""
        m_id = raw_tx["merchant_id"]
        p_id = raw_tx.get("payer_id", f"payer_{uuid.uuid4().hex[:6]}")
        mcc_str = str(raw_tx["mcc_raw"])
        raw_amount = float(raw_tx["raw_amount"])
        amount = raw_amount / 100.0 if raw_amount > 500 else raw_amount

        try:
            ts_str = str(raw_tx.get("timestamp", ""))
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        except Exception:
            dt = datetime.utcnow()
        unix_ts = dt.timestamp()

        # 1. Location & Impossible Travel
        loc_info = self.location_tracker.evaluate_location(
            terminal_id=raw_tx.get("terminal_id", ""),
            m_lat=raw_tx.get("merchant_lat"),
            m_lon=raw_tx.get("merchant_lon"),
            p_lat=raw_tx.get("payer_lat"),
            p_lon=raw_tx.get("payer_lon"),
            current_ts=unix_ts,
        )

        # 2. Live Category Macro-Surge
        surge_mult = self.category_tracker.update_and_get_surge(mcc_str, m_id, unix_ts)

        # 3. Velocity & Unique Payer Diversity
        one_hour_ago = unix_ts - 3600.0
        act_queue = self.hourly_activity[m_id]
        while act_queue and act_queue[0][0] < one_hour_ago:
            act_queue.popleft()
        act_queue.append((unix_ts, p_id))

        velocity_1hr = len(act_queue)
        unique_payers_1hr = len({item[1] for item in act_queue})
        unique_payer_ratio = unique_payers_1hr / max(1, velocity_1hr)
        effective_seasonal_scale = surge_mult if unique_payer_ratio >= 0.65 else 1.0

        # 4. Temporal Anomaly & Adaptive Velocity Scores
        hist_queue = self.rolling_history[m_id]
        raw_mse, anomaly_score = self._compute_temporal_scores(hist_queue, dt, effective_seasonal_scale)
        hist_queue.append((amount, dt))

        velocity_score = self._compute_velocity_score(
            velocity_1hr=velocity_1hr,
            unique_payer_ratio=unique_payer_ratio,
            category_surge_multiplier=effective_seasonal_scale,
        )

        # 5. Dynamic GNN Node Update & Syndicate Score
        preliminary_risk = max(anomaly_score, velocity_score)
        gnn_vector, is_new_merchant, syndicate_score = self.graph.register_and_update_node(
            m_id=m_id,
            p_id=p_id,
            is_flagged_risky=(preliminary_risk > 0.65),
            risk_signal=preliminary_risk,
        )

        # 6. Tabular Feature Assembly & CatBoost Prediction
        hour_float = dt.hour + (dt.minute / 60.0)
        hour_rad = hour_float * (2.0 * math.pi / 24.0)
        hardware_str = str(raw_tx.get("terminal_hardware_type", "")).upper()
        is_soundbox = 1 if ("SOUNDBOX" in hardware_str or "POS_TERMINAL" in hardware_str) else 0

        calibrated_velocity = (
            int(velocity_1hr / effective_seasonal_scale)
            if (effective_seasonal_scale > 1.2 and is_soundbox == 1)
            else velocity_1hr
        )

        tx_features = {
            "amount": amount,
            "is_soundbox": is_soundbox,
            "mcc": mcc_str,
            "is_weekend": 1 if dt.weekday() >= 5 else 0,
            "is_round_amount": 1 if (amount > 0 and amount % 10.0 == 0) else 0,
            "hour_sin": float(math.sin(hour_rad)),
            "hour_cos": float(math.cos(hour_rad)),
            "merchant_1hr_velocity": calibrated_velocity,
            "temporal_anomaly_score": raw_mse,
        }

        feature_row = gnn_vector.tolist() + [tx_features[col] for col in self.tabular_cols]
        overall_prob = float(self.cat_model.predict_proba([feature_row], thread_count=1)[0][1])

        if loc_info["is_terminal_cloned"]:
            overall_prob = max(overall_prob, 0.99)
        elif is_soundbox == 1 and loc_info["payer_to_merchant_km"] > 500.0:
            overall_prob = max(overall_prob, 0.75)

        if overall_prob > 0.70:
            _, _, syndicate_score = self.graph.register_and_update_node(
                m_id=m_id, p_id=p_id, is_flagged_risky=True, risk_signal=overall_prob
            )

        action = "BLOCK" if overall_prob > 0.80 else ("REVIEW" if overall_prob > 0.50 else "ALLOW")
        explanations = self._build_shap_explanations(
            feature_row,
            syndicate_score,
            anomaly_score,
            velocity_score,
            raw_mse,
            velocity_1hr,
            unique_payer_ratio,
            surge_mult,
            loc_info,
            tx_features,
        )

        result = {
            "event_id": raw_tx.get("raw_event_id"),
            "timestamp": raw_tx.get("timestamp"),
            "merchant_id": m_id,
            "is_new_merchant_node": is_new_merchant,
            "amount": round(amount, 2),
            "scores": {
                "overall_fraud_score": round(overall_prob, 4),
                "syndicate_score": syndicate_score,
                "anomaly_transaction_score": anomaly_score,
                "velocity_score": velocity_score,
            },
            "location": loc_info,
            "seasonal_context": {
                "mcc_surge_multiplier": surge_mult,
                "unique_payer_ratio": round(unique_payer_ratio, 2),
            },
            "action": action,
            "shap_explanations": explanations,
        }

        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**raw_tx, **result}) + "\n")

        if self.print_terminal:
            self._print_terminal(result)
        return result

    @staticmethod
    def _print_terminal(res: dict):
        s = res["scores"]
        new_tag = " [NEW NODE]" if res["is_new_merchant_node"] else ""
        print("=" * 80)
        print(f"EVENT: {res['event_id']} | MERCHANT: {res['merchant_id']}{new_tag} | ACTION: {res['action']}")
        print(
            f"SCORES -> Fraud: {s['overall_fraud_score']:.4f} | Syndicate: {s['syndicate_score']:.4f} | "
            f"Anomaly: {s['anomaly_transaction_score']:.4f} | Velocity: {s['velocity_score']:.4f}"
        )
        for exp in res["shap_explanations"]:
            print(f"  {exp}")
        print("=" * 80)

    def capture_live_file(self, watch_file: str = "live_incoming_tx.jsonl"):
        """Continuously tails a live transaction JSONL stream file and evaluates events in real time."""
        print(f"[LIVE CAPTURE] Watching '{watch_file}' for incoming live transactions... (Ctrl+C to stop)\n")
        if not os.path.exists(watch_file):
            open(watch_file, "w", encoding="utf-8").close()

        with open(watch_file, "r", encoding="utf-8") as f:
            f.seek(0, os.SEEK_END)
            try:
                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(0.05)
                        continue
                    line = line.strip()
                    if line:
                        self.evaluate(json.loads(line))
            except KeyboardInterrupt:
                print("\n[LIVE CAPTURE] Stopped. Saving updated GNN merchant graph...")
                self.graph.save()

    def capture_live_stdin(self):
        """Captures live JSON transactions piped via STDIN."""
        print("[LIVE CAPTURE] Listening on STDIN for live JSON transactions...\n")
        try:
            for line in sys.stdin:
                line = line.strip()
                if line:
                    self.evaluate(json.loads(line))
        except KeyboardInterrupt:
            pass
        finally:
            self.graph.save()


if __name__ == "__main__":
    engine = FraudEngine()
    if "--stdin" in sys.argv:
        engine.capture_live_stdin()
    else:
        engine.capture_live_file("live_incoming_tx.jsonl")