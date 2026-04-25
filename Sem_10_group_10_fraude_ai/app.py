"""
app.py
------
Fraud Detection Web Application – Flask entry point.

Routes
------
GET  /          → render transaction input form
POST /predict   → run hybrid fraud detection, return JSON result

Run
---
    python create_model.py   # only once – builds fraud_model.pkl
    python app.py
"""

import os
import json
import logging
from io import BytesIO
from datetime import datetime

import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from feature_engineering import extract_features, rules_based_score

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=".")

# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "fraud_model.pkl")

def load_model():
    """Load or auto-create the model bundle."""
    if not os.path.exists(MODEL_PATH):
        log.warning("Model not found – auto-creating …")
        import create_model
        create_model.build_and_save_model()

    bundle = joblib.load(MODEL_PATH)
    log.info("Model bundle loaded  (version %s)", bundle.get("version", "?"))
    return bundle

MODEL_BUNDLE = load_model()


# ─────────────────────────────────────────────────────────────
# REQUEST / REPORT HELPERS
# ─────────────────────────────────────────────────────────────
def parse_request_payload() -> dict:
    """Parse incoming request data and normalize booleans."""
    if request.is_json:
        data = request.get_json(force=True)
    else:
        data = request.form.to_dict()

    data["is_new_device"] = data.get("is_new_device") in (
        True, "true", "True", "on", "1", 1
    )
    return data


def build_pdf_report(form_data: dict, result: dict) -> BytesIO:
    """Generate a themed FraudGuard AI PDF report and return in-memory bytes."""
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    def set_fill(rgb: tuple[float, float, float]) -> None:
        pdf.setFillColorRGB(*rgb)

    def set_stroke(rgb: tuple[float, float, float]) -> None:
        pdf.setStrokeColorRGB(*rgb)

    def metric_color(score: float) -> tuple[float, float, float]:
        if score >= 60:
            return (0.94, 0.31, 0.29)  # red
        if score >= 35:
            return (0.96, 0.67, 0.14)  # amber
        return (0.19, 0.84, 0.49)      # green

    def draw_progress(label: str, value: float) -> None:
        nonlocal y
        bar_w = 210
        bar_h = 10
        x = 70
        set_fill((0.63, 0.67, 0.78))
        pdf.setFont("Helvetica", 10)
        pdf.drawString(x, y, label)
        y -= 12

        set_fill((0.16, 0.18, 0.24))
        pdf.roundRect(x, y - bar_h, bar_w, bar_h, 4, stroke=0, fill=1)
        fill_w = max(0, min(100, float(value))) / 100 * bar_w
        set_fill(metric_color(float(value)))
        pdf.roundRect(x, y - bar_h, fill_w, bar_h, 4, stroke=0, fill=1)

        set_fill((0.89, 0.91, 0.96))
        pdf.drawRightString(x + bar_w + 40, y - bar_h + 1, f"{value:.1f}")
        y -= 22

    def draw_section_title(title: str) -> None:
        nonlocal y
        set_fill((0.29, 0.94, 0.77))
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, title)
        y -= 16

    pdf.setTitle("FraudGuard AI Transaction Report")

    # Background
    set_fill((0.04, 0.05, 0.07))
    pdf.rect(0, 0, width, height, stroke=0, fill=1)

    # Header strip
    set_fill((0.08, 0.10, 0.14))
    pdf.rect(0, height - 88, width, 88, stroke=0, fill=1)
    set_stroke((0.15, 0.17, 0.22))
    pdf.setLineWidth(1)
    pdf.line(0, height - 88, width, height - 88)

    set_fill((0.29, 0.94, 0.77))
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(48, height - 55, "FraudGuard AI")
    set_fill((0.89, 0.91, 0.96))
    pdf.setFont("Helvetica", 11)
    pdf.drawString(48, height - 72, "Transaction Risk Report")
    pdf.drawRightString(width - 45, height - 72, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    y = height - 120

    # Summary card
    set_fill((0.08, 0.10, 0.14))
    pdf.roundRect(45, y - 88, width - 90, 88, 10, stroke=0, fill=1)
    set_stroke((0.19, 0.22, 0.30))
    pdf.roundRect(45, y - 88, width - 90, 88, 10, stroke=1, fill=0)

    score = float(result["fraud_score"])
    score_color = metric_color(score)
    set_fill((0.63, 0.67, 0.78))
    pdf.setFont("Helvetica", 10)
    pdf.drawString(62, y - 24, "Fraud Score")
    set_fill(score_color)
    pdf.setFont("Helvetica-Bold", 34)
    pdf.drawString(62, y - 58, f"{score:.1f}")
    set_fill((0.63, 0.67, 0.78))
    pdf.setFont("Helvetica", 12)
    pdf.drawString(130, y - 55, "/ 100")

    set_fill((0.89, 0.91, 0.96))
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(250, y - 28, f"Verdict: {result['verdict']}")
    pdf.drawString(250, y - 50, f"Risk: {result['risk_level']}")

    y -= 110

    draw_section_title("Signal Breakdown")
    draw_progress("Isolation Forest", float(result["breakdown"]["isolation_forest"]))
    draw_progress("Local Outlier Factor", float(result["breakdown"]["lof"]))
    draw_progress("Rules Engine", float(result["breakdown"]["rules"]))

    y -= 4
    draw_section_title("Risk Indicators")
    set_fill((0.89, 0.91, 0.96))
    pdf.setFont("Helvetica", 10)
    for reason in result["reasons"]:
        if y < 85:
            pdf.showPage()
            set_fill((0.04, 0.05, 0.07))
            pdf.rect(0, 0, width, height, stroke=0, fill=1)
            y = height - 60
            draw_section_title("Risk Indicators (continued)")
            set_fill((0.89, 0.91, 0.96))
            pdf.setFont("Helvetica", 10)
        pdf.drawString(64, y, f"- {reason}")
        y -= 15

    y -= 8
    if y < 155:
        pdf.showPage()
        set_fill((0.04, 0.05, 0.07))
        pdf.rect(0, 0, width, height, stroke=0, fill=1)
        y = height - 60

    draw_section_title("Input Transaction")
    set_fill((0.89, 0.91, 0.96))
    pdf.setFont("Helvetica", 10)
    ordered_keys = [
        "amount",
        "transaction_type",
        "location",
        "transaction_hour",
        "device_type",
        "transaction_freq",
        "account_age_days",
        "failed_attempts",
        "is_new_device",
    ]
    for key in ordered_keys:
        if key in form_data:
            if y < 60:
                pdf.showPage()
                set_fill((0.04, 0.05, 0.07))
                pdf.rect(0, 0, width, height, stroke=0, fill=1)
                y = height - 60
                draw_section_title("Input Transaction (continued)")
                set_fill((0.89, 0.91, 0.96))
                pdf.setFont("Helvetica", 10)

            label = key.replace("_", " ").title()
            value = str(form_data.get(key))
            pdf.drawString(64, y, f"{label}:")
            pdf.drawRightString(width - 60, y, value)
            y -= 14

    pdf.save()
    buffer.seek(0)
    return buffer


# ─────────────────────────────────────────────────────────────
# EXPLANATION HELPERS
# ─────────────────────────────────────────────────────────────
def ml_explanations(form_data: dict, iso_score: float, lof_score: float) -> list[str]:
    """
    Create grounded, deterministic explanations when ML flags risk
    but rules-based reasons are sparse.
    """
    reasons = []

    if iso_score >= 70:
        reasons.append("Isolation Forest detected a strong outlier pattern")
    elif iso_score >= 55:
        reasons.append("Isolation Forest detected a moderate outlier pattern")

    if lof_score >= 70:
        reasons.append("Local Outlier Factor found this transaction unlike nearby behavior")
    elif lof_score >= 55:
        reasons.append("Local Outlier Factor found partial neighborhood deviation")

    # Add concrete context from submitted values (avoids generic fallback text).
    amount = float(form_data.get("amount", 0))
    tx_hour = int(form_data.get("transaction_hour", 12))
    tx_freq = int(form_data.get("transaction_freq", 1))
    account_age = int(form_data.get("account_age_days", 365))
    failed_attempts = int(form_data.get("failed_attempts", 0))
    location = str(form_data.get("location", "domestic")).lower()
    tx_type = str(form_data.get("transaction_type", "")).lower()
    is_new_device = bool(form_data.get("is_new_device", False))

    if amount >= 5000:
        reasons.append(f"Large payment amount observed (${amount:,.2f})")
    if tx_hour <= 5:
        reasons.append(f"Executed during odd hours ({tx_hour:02d}:00)")
    if tx_freq >= 10:
        reasons.append(f"High recent transaction velocity ({tx_freq} transactions)")
    if account_age < 30:
        reasons.append(f"Very new account profile ({account_age} days)")
    if failed_attempts >= 2:
        reasons.append(f"Recent authentication failures noted ({failed_attempts})")
    if location in ("international", "high_risk", "vpn", "tor"):
        reasons.append(f"Higher-risk location category ({location})")
    if tx_type == "crypto_transfer":
        reasons.append("Crypto transfer category carries elevated baseline risk")
    if is_new_device:
        reasons.append("Transaction initiated from an unrecognized device")

    # Keep order while removing duplicates.
    unique = list(dict.fromkeys(reasons))
    return unique[:4]


def risk_reduction_advice(form_data: dict) -> list[str]:
    """
    Provide deterministic what-if suggestions to reduce fraud risk.
    This is an action-oriented helper, separate from model verdicting.
    """
    advice = []

    amount = float(form_data.get("amount", 0))
    tx_hour = int(form_data.get("transaction_hour", 12))
    tx_freq = int(form_data.get("transaction_freq", 1))
    account_age = int(form_data.get("account_age_days", 365))
    failed_attempts = int(form_data.get("failed_attempts", 0))
    location = str(form_data.get("location", "domestic")).lower()
    tx_type = str(form_data.get("transaction_type", "")).lower()
    is_new_device = bool(form_data.get("is_new_device", False))

    if location in ("tor", "vpn", "high_risk", "international"):
        advice.append("Route through trusted network/location profile (expected risk drop: high)")
    if tx_hour <= 5:
        advice.append("Delay transaction to regular business hours (expected risk drop: medium)")
    if amount > 5000:
        advice.append("Split into smaller verified transactions when operationally valid (expected risk drop: medium)")
    if failed_attempts >= 2:
        advice.append("Reset authentication and re-verify identity before retry (expected risk drop: high)")
    if tx_freq >= 10:
        advice.append("Reduce transaction burst/velocity and stagger payments (expected risk drop: medium)")
    if is_new_device:
        advice.append("Complete step-up verification for this new device first (expected risk drop: medium)")
    if account_age < 30:
        advice.append("Apply manual review hold for very new accounts (expected risk drop: medium)")
    if tx_type == "crypto_transfer":
        advice.append("Add destination whitelist + confirmation checks for crypto transfers (expected risk drop: medium)")

    if not advice:
        advice.append("No immediate mitigation needed; keep monitoring and standard verification controls")

    return advice[:4]


# ─────────────────────────────────────────────────────────────
# HYBRID SCORING
# ─────────────────────────────────────────────────────────────
def compute_fraud_score(form_data: dict) -> dict:
    """
    Combine three signals into one 0-100 fraud risk score.

    Signal weights
    --------------
    - IsolationForest score  : 40 %
    - LOF score              : 20 %
    - Rules-based score      : 40 %
    """
    # ── Feature vector ──────────────────────────────────────────
    X = extract_features(form_data)          # shape (1, 10)

    # ── IsolationForest  (score_samples → more-negative = more anomalous) ──
    iso  = MODEL_BUNDLE["isolation_forest"]
    raw_iso = iso.score_samples(X)[0]        # typically −0.7 … −0.3 for normal

    # Map to 0-100 (clamp raw range [−0.8, −0.2])
    iso_score = float(np.clip((raw_iso + 0.8) / 0.6, 0, 1))
    iso_score = (1 - iso_score) * 100        # invert: higher = more fraudulent

    # ── Local Outlier Factor ─────────────────────────────────────
    lof = MODEL_BUNDLE["lof"]
    raw_lof = lof.score_samples(X)[0]        # negative_outlier_factor style
    lof_score = float(np.clip((raw_lof + 2.0) / 2.0, 0, 1))
    lof_score = (1 - lof_score) * 100

    # ── Rules-based score ────────────────────────────────────────
    rule_score, reasons = rules_based_score(form_data)

    # ── Weighted hybrid score ────────────────────────────────────
    final_score = (
        0.40 * iso_score  +
        0.20 * lof_score  +
        0.40 * rule_score
    )
    final_score = round(float(np.clip(final_score, 0, 100)), 1)

    # ── Verdict ──────────────────────────────────────────────────
    if final_score >= 60:
        verdict    = "FRAUDULENT"
        risk_level = "HIGH"
    elif final_score >= 45:
        verdict    = "SUSPICIOUS"
        risk_level = "MEDIUM"
    else:
        verdict    = "LEGITIMATE"
        risk_level = "LOW"

    # Build grounded reasons for ML-driven suspicious cases.
    if verdict in ("FRAUDULENT", "SUSPICIOUS"):
        ml_reasons = ml_explanations(form_data, iso_score, lof_score)
        combined_reasons = list(dict.fromkeys([*reasons, *ml_reasons]))
        reasons = combined_reasons[:5]

    # Final fallback for low-risk cases with no indicators.
    if not reasons:
        reasons = ["No fraud indicators detected"]

    advice = risk_reduction_advice(form_data)

    return {
        "fraud_score":  final_score,
        "verdict":      verdict,
        "risk_level":   risk_level,
        "reasons":      reasons,
        "advice":       advice,
        "breakdown": {
            "isolation_forest": round(iso_score,  1),
            "lof":              round(lof_score,  1),
            "rules":            round(rule_score, 1),
        },
    }


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Render the transaction input form."""
    return render_template("index.html")


@app.route("/style.css")
def style_css():
    """Serve CSS file from project root."""
    return send_from_directory(".", "style.css")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept form data or JSON, run hybrid fraud detection,
    and return a JSON result.
    """
    try:
        data = parse_request_payload()
        log.info("Incoming transaction: %s", json.dumps(data))

        result = compute_fraud_score(data)
        log.info(
            "Result → %s  score=%.1f",
            result["verdict"], result["fraud_score"]
        )
        return jsonify({"success": True, **result})

    except (ValueError, KeyError, TypeError) as exc:
        log.exception("Prediction error")
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/report", methods=["POST"])
def report():
    """Generate and download a PDF report for a transaction."""
    try:
        data = parse_request_payload()
        result = compute_fraud_score(data)
        pdf_buffer = build_pdf_report(data, result)
        filename = f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype="application/pdf",
        )
    except (ValueError, KeyError, TypeError) as exc:
        log.exception("Report generation error")
        return jsonify({"success": False, "error": str(exc)}), 400


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Fraud Detection API  –  http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=True, host="localhost", port=5000)
