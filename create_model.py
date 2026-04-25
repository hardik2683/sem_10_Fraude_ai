"""
create_model.py
---------------
Initialises and serialises the hybrid fraud-detection model.

Run once before starting the Flask server:
    python create_model.py

This script:
  1. Generates synthetic "normal" transaction data to fit IsolationForest
     (no real labelled dataset required).
  2. Fits a Local Outlier Factor detector on the same data.
  3. Bundles both models + metadata into model/fraud_model.pkl.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join("model", "fraud_model.pkl")
RANDOM_STATE  = 42
N_SAMPLES     = 2_000   # synthetic "normal" transactions for fitting
CONTAMINATION = 0.05    # expected fraud rate ≈ 5 %


def generate_normal_transactions(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Produce synthetic feature vectors that represent *normal* transactions.
    Feature layout (matches feature_engineering.extract_features):
        0  amount_n         – normalised amount          [0, 1]
        1  tx_type_enc      – encoded transaction type   [0, 1]
        2  location_risk    – location risk score        [0, 1]
        3  hour_n           – normalised hour            [0, 1]
        4  device_enc       – encoded device             [0, 1]
        5  freq_n           – normalised frequency       [0, 1]
        6  age_n            – normalised account age     [0, 1]
        7  failed_n         – normalised failed attempts [0, 1]
        8  is_new_device    – 0 / 1 flag
        9  odd_hour         – 0 / 1 flag
    """
    data = np.column_stack([
        rng.uniform(0.00, 0.30, n),          # 0  low-to-medium amounts
        rng.choice([0, 0.25, 0.75, 1], n),   # 1  common tx types
        rng.uniform(0.05, 0.30, n),          # 2  low location risk
        rng.uniform(0.25, 0.85, n),          # 3  daytime hours
        rng.choice([0, 0.25, 0.5], n),       # 4  known devices
        rng.uniform(0.02, 0.20, n),          # 5  low frequency
        rng.uniform(0.20, 1.00, n),          # 6  established accounts
        rng.uniform(0.00, 0.10, n),          # 7  few failed attempts
        rng.integers(0, 2, n),               # 8  new-device flag
        rng.choice([0, 1], n, p=[0.85, 0.15]),  # 9  odd-hour (15 % chance)
    ])
    return data


def build_and_save_model() -> None:
    rng = np.random.default_rng(RANDOM_STATE)

    print("[1/4] Generating synthetic normal-transaction data …")
    X_normal = generate_normal_transactions(N_SAMPLES, rng)

    # ── IsolationForest ─────────────────────────────────────────
    print("[2/4] Fitting IsolationForest …")
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso_forest.fit(X_normal)

    # ── Local Outlier Factor (novelty=True → supports .predict) ─
    print("[3/4] Fitting Local Outlier Factor …")
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=CONTAMINATION,
        novelty=True,
        n_jobs=-1,
    )
    lof.fit(X_normal)

    # ── Bundle & persist ────────────────────────────────────────
    print("[4/4] Saving model bundle …")
    os.makedirs("model", exist_ok=True)
    bundle = {
        "isolation_forest": iso_forest,
        "lof":              lof,
        "feature_names": [
            "amount_n", "tx_type_enc", "location_risk", "hour_n",
            "device_enc", "freq_n", "age_n", "failed_n",
            "is_new_device", "odd_hour",
        ],
        "version": "1.0.0",
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"    ✓  Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    build_and_save_model()
    print("\nDone. Run  python app.py  to start the web server.")
