"""
feature_engineering.py
-----------------------
Handles all feature extraction, encoding, and normalization
for the fraud detection pipeline.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# ENCODING MAPS  (label → integer)
# ─────────────────────────────────────────────
TRANSACTION_TYPE_MAP = {
    "online_purchase": 0,
    "atm_withdrawal":  1,
    "bank_transfer":   2,
    "pos_payment":     3,
    "crypto_transfer": 4,
}

DEVICE_TYPE_MAP = {
    "mobile":  0,
    "desktop": 1,
    "tablet":  2,
    "atm":     3,
    "unknown": 4,
}

LOCATION_RISK_MAP = {
    # Low-risk domestic
    "local":         0.1,
    "domestic":      0.2,
    # Medium-risk
    "international": 0.6,
    # High-risk categories
    "high_risk":     0.9,
    "vpn":           0.85,
    "tor":           1.0,
}

# ─────────────────────────────────────────────
# FEATURE BOUNDS  (used for min-max normalisation)
# ─────────────────────────────────────────────
FEATURE_BOUNDS = {
    "amount":            (0,    50_000),
    "transaction_hour":  (0,    23),
    "transaction_freq":  (1,    50),
    "account_age_days":  (1,    3650),
    "failed_attempts":   (0,    10),
}


def encode_transaction_type(tx_type: str) -> int:
    """Return integer code for a transaction type (default 0)."""
    return TRANSACTION_TYPE_MAP.get(tx_type.lower(), 0)


def encode_device(device: str) -> int:
    """Return integer code for a device type (default 4 = unknown)."""
    return DEVICE_TYPE_MAP.get(device.lower(), 4)


def encode_location(location: str) -> float:
    """Return a 0-1 risk score for a location category."""
    return LOCATION_RISK_MAP.get(location.lower(), 0.5)


def minmax_scale(value: float, lo: float, hi: float) -> float:
    """Clip then scale a single value to [0, 1]."""
    value = max(lo, min(hi, value))
    return (value - lo) / (hi - lo) if hi > lo else 0.0


def extract_features(form_data: dict) -> np.ndarray:
    """
    Convert raw form input into a normalised feature vector.

    Parameters
    ----------
    form_data : dict
        Keys expected:
            amount, transaction_type, location, transaction_hour,
            device_type, transaction_freq, account_age_days,
            failed_attempts, is_new_device (bool)

    Returns
    -------
    np.ndarray  shape (1, 10)
    """

    # ── Raw values ──────────────────────────────────────────────
    amount           = float(form_data.get("amount", 0))
    tx_type          = str(form_data.get("transaction_type", "online_purchase"))
    location         = str(form_data.get("location", "domestic"))
    tx_hour          = int(form_data.get("transaction_hour", 12))
    device           = str(form_data.get("device_type", "mobile"))
    tx_freq          = int(form_data.get("transaction_freq", 1))
    account_age      = int(form_data.get("account_age_days", 365))
    failed_attempts  = int(form_data.get("failed_attempts", 0))
    is_new_device    = int(bool(form_data.get("is_new_device", False)))

    # ── Derived feature: odd-hour flag (midnight-5am) ────────────
    odd_hour = 1 if tx_hour in range(0, 6) else 0

    # ── Encode categoricals ──────────────────────────────────────
    tx_type_enc   = encode_transaction_type(tx_type) / max(len(TRANSACTION_TYPE_MAP) - 1, 1)
    device_enc    = encode_device(device)             / max(len(DEVICE_TYPE_MAP) - 1,     1)
    location_risk = encode_location(location)

    # ── Normalise numerics ───────────────────────────────────────
    b = FEATURE_BOUNDS
    amount_n    = minmax_scale(amount,          *b["amount"])
    hour_n      = minmax_scale(tx_hour,         *b["transaction_hour"])
    freq_n      = minmax_scale(tx_freq,         *b["transaction_freq"])
    age_n       = minmax_scale(account_age,     *b["account_age_days"])
    failed_n    = minmax_scale(failed_attempts, *b["failed_attempts"])

    # ── Assemble vector ──────────────────────────────────────────
    features = np.array([[
        amount_n,       # 0 – transaction amount
        tx_type_enc,    # 1 – transaction type
        location_risk,  # 2 – location risk score
        hour_n,         # 3 – hour of day
        device_enc,     # 4 – device type
        freq_n,         # 5 – transaction frequency
        age_n,          # 6 – account age
        failed_n,       # 7 – recent failed attempts
        is_new_device,  # 8 – new device flag
        odd_hour,       # 9 – odd-hour flag
    ]])

    return features  # shape (1, 10)


# ─────────────────────────────────────────────
# RULES-BASED SCORER
# ─────────────────────────────────────────────
def rules_based_score(form_data: dict) -> tuple[float, list[str]]:
    """
    Apply deterministic fraud-indicator rules.

    Returns
    -------
    score   : float  0-100 risk points from rules alone
    reasons : list[str]  human-readable explanations
    """
    score   = 0.0
    reasons = []

    amount          = float(form_data.get("amount", 0))
    tx_hour         = int(form_data.get("transaction_hour", 12))
    location        = str(form_data.get("location", "domestic")).lower()
    failed_attempts = int(form_data.get("failed_attempts", 0))
    tx_freq         = int(form_data.get("transaction_freq", 1))
    account_age     = int(form_data.get("account_age_days", 365))
    is_new_device   = bool(form_data.get("is_new_device", False))
    tx_type         = str(form_data.get("transaction_type", "")).lower()

    # Rule 1 – Very high amount
    if amount > 10_000:
        score += 25
        reasons.append(f"High transaction amount (${amount:,.2f})")
    elif amount > 5_000:
        score += 12
        reasons.append(f"Elevated transaction amount (${amount:,.2f})")

    # Rule 2 – Odd-hours (midnight → 5 am)
    if 0 <= tx_hour <= 5:
        score += 15
        reasons.append(f"Transaction at unusual hour ({tx_hour:02d}:00)")

    # Rule 3 – High-risk or anonymising locations
    if location in ("tor", "vpn"):
        score += 30
        reasons.append(f"Transaction routed via anonymising network ({location.upper()})")
    elif location == "high_risk":
        score += 20
        reasons.append("Transaction from high-risk geographic region")
    elif location == "international":
        score += 10
        reasons.append("International transaction detected")

    # Rule 4 – Multiple failed authentication attempts
    if failed_attempts >= 5:
        score += 25
        reasons.append(f"Multiple failed login attempts ({failed_attempts})")
    elif failed_attempts >= 2:
        score += 10
        reasons.append(f"Recent failed login attempts ({failed_attempts})")

    # Rule 5 – Unusually high transaction frequency
    if tx_freq >= 20:
        score += 20
        reasons.append(f"Very high transaction frequency ({tx_freq} recent transactions)")
    elif tx_freq >= 10:
        score += 10
        reasons.append(f"High transaction frequency ({tx_freq} recent transactions)")

    # Rule 6 – Brand-new account
    if account_age < 30:
        score += 15
        reasons.append(f"Very new account (only {account_age} days old)")

    # Rule 7 – New / unrecognised device
    if is_new_device:
        score += 10
        reasons.append("Transaction from an unrecognised device")

    # Rule 8 – Risky transaction type combos
    if tx_type == "crypto_transfer" and amount > 1_000:
        score += 20
        reasons.append("Large crypto transfer — high-risk category")
    elif tx_type == "bank_transfer" and amount > 5_000:
        score += 10
        reasons.append("Large bank transfer flagged for review")

    return min(score, 100.0), reasons
