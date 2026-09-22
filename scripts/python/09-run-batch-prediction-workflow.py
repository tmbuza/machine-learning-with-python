"""Run a validated, traceable batch-scoring workflow for MLPY-009."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["tenure_months", "monthly_charge", "support_contacts", "usage_hours", "plan", "region"]
NUMERIC = FEATURES[:4]
CATEGORICAL = FEATURES[4:]
MODEL_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"
THRESHOLD = 0.55
BATCH_ID = "retention-2026-07"
RUN_ID = "run-09-reproducible-example"
TABLE_DIR = ROOT / "results" / "tables"
LOG_PATH = ROOT / "results" / "logs" / "09-batch-scoring.jsonl"
MANIFEST_PATH = ROOT / "results" / "manifests" / "09-run-manifest.json"
FIGURE_PATH = ROOT / "results" / "figures" / "09-batch-prediction-operations.png"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def log_event(events, event, severity="INFO", **details):
    events.append({"timestamp_utc": now(), "run_id": RUN_ID, "batch_id": BATCH_ID, "event": event, "severity": severity, **details})


def create_training_and_batch():
    rng = np.random.default_rng(SEED)
    count = 520
    plans = rng.choice(["Basic", "Standard", "Premium"], count, p=[0.42, 0.38, 0.20])
    regions = rng.choice(["North", "South", "East", "West"], count)
    risk = rng.normal(0, 0.75, count)
    usage_base = rng.normal(43, 9, count)
    rows = []
    for month, date in enumerate(pd.date_range("2026-01-31", periods=6, freq="ME")):
        for position in range(count):
            tenure = int(rng.integers(2, 70) + month)
            contacts = int(rng.poisson(1.1 + 0.45 * max(risk[position], 0)))
            usage = usage_base[position] + {"Basic": -7, "Standard": 1, "Premium": 9}[plans[position]] + month + rng.normal(0, 5)
            charge = {"Basic": 35, "Standard": 65, "Premium": 95}[plans[position]] + rng.normal(0, 5)
            logit = -1.4 + risk[position] + 0.55 * contacts - 0.02 * tenure - 0.043 * usage + 0.08 * month + (0.38 if plans[position] == "Basic" else 0)
            probability = 1 / (1 + np.exp(-logit))
            rows.append({"customer_id": f"C{position + 1:04d}", "snapshot_date": date, "tenure_months": tenure, "monthly_charge": round(float(charge), 2), "support_contacts": contacts, "usage_hours": round(max(float(usage), 0), 2), "plan": plans[position], "region": regions[position], "churned_next_month": int(rng.random() < probability)})
    training = pd.DataFrame(rows)
    training.loc[rng.choice(training.index, 55, replace=False), "monthly_charge"] = np.nan
    batch = training[training["snapshot_date"] == training["snapshot_date"].max()].sample(200, random_state=SEED).drop(columns=["snapshot_date", "churned_next_month"]).reset_index(drop=True)
    batch.insert(1, "prediction_date", "2026-07-31")
    batch.loc[:7, "monthly_charge"] = np.nan
    batch.loc[0, "customer_id"] = None
    batch.loc[1, "tenure_months"] = -4
    batch.loc[2, "plan"] = "Legacy"
    batch.loc[3, "region"] = "Unknown-region"
    batch.loc[4, "support_contacts"] = -1
    batch.loc[5, "prediction_date"] = "not-a-date"
    batch.loc[6, "customer_id"] = batch.loc[7, "customer_id"]
    return training, batch


def build_pipeline():
    preprocessing = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
    ])
    return make_pipeline(preprocessing, LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))


def validate_rows(data):
    validation = pd.DataFrame(index=data.index)
    validation["missing_customer_id"] = data["customer_id"].isna() | data["customer_id"].astype(str).str.strip().eq("")
    parsed_dates = pd.to_datetime(data["prediction_date"], errors="coerce")
    validation["invalid_prediction_date"] = parsed_dates.isna() | parsed_dates.ne(pd.Timestamp("2026-07-31"))
    validation["duplicate_customer_date"] = data.assign(_date=parsed_dates).duplicated(["customer_id", "_date"], keep=False)
    validation["invalid_tenure"] = pd.to_numeric(data["tenure_months"], errors="coerce").lt(0) | pd.to_numeric(data["tenure_months"], errors="coerce").isna()
    validation["invalid_support_contacts"] = pd.to_numeric(data["support_contacts"], errors="coerce").lt(0) | pd.to_numeric(data["support_contacts"], errors="coerce").isna()
    validation["invalid_plan"] = ~data["plan"].isin(["Basic", "Standard", "Premium"])
    validation["invalid_region"] = ~data["region"].isin(["North", "South", "East", "West"])
    rule_columns = list(validation.columns)
    validation["accepted"] = ~validation[rule_columns].any(axis=1)
    validation["failure_reasons"] = validation[rule_columns].apply(lambda row: ";".join(row.index[row].tolist()), axis=1)
    return validation


def prediction_id(row):
    key = f"{row.customer_id}|{row.prediction_date.date().isoformat()}|{MODEL_VERSION}"
    return sha256(key.encode("utf-8")).hexdigest()


def main():
    started = time.perf_counter()
    events = []
    log_event(events, "run_started", model_version=MODEL_VERSION, schema_version=SCHEMA_VERSION)
    for path in [ROOT / "data" / "processed", ROOT / "data" / "scoring", ROOT / "data" / "quarantine", TABLE_DIR, LOG_PATH.parent, MANIFEST_PATH.parent, FIGURE_PATH.parent]:
        path.mkdir(parents=True, exist_ok=True)
    training, batch = create_training_and_batch()
    training_path = ROOT / "data" / "processed" / "09-training-data.csv"
    batch_path = ROOT / "data" / "scoring" / "09-input-batch.csv"
    training.to_csv(training_path, index=False, date_format="%Y-%m-%d")
    batch.to_csv(batch_path, index=False)
    log_event(events, "input_loaded", total_rows=len(batch), input_sha256=digest(batch_path))

    required = set(FEATURES + ["customer_id", "prediction_date"])
    missing_columns = sorted(required - set(batch.columns))
    if missing_columns:
        log_event(events, "batch_failed", severity="ERROR", missing_columns=missing_columns)
        raise ValueError(f"Missing required columns: {missing_columns}")
    validation = validate_rows(batch)
    validation.insert(0, "source_row", validation.index + 2)
    validation.to_csv(TABLE_DIR / "09-row-validation.csv", index=False)
    rejected = batch.loc[~validation["accepted"]].copy()
    rejected["failure_reasons"] = validation.loc[~validation["accepted"], "failure_reasons"].to_numpy()
    rejected.to_csv(ROOT / "data" / "quarantine" / "09-rejected-rows.csv", index=False)
    accepted = batch.loc[validation["accepted"]].copy()
    accepted["prediction_date"] = pd.to_datetime(accepted["prediction_date"])
    log_event(events, "validation_completed", accepted_rows=len(accepted), rejected_rows=len(rejected))

    pipeline = build_pipeline()
    pipeline.fit(training[FEATURES], training["churned_next_month"])
    probability = pipeline.predict_proba(accepted[FEATURES])[:, 1]
    output = accepted[["customer_id", "prediction_date", "plan", "region"]].copy()
    output["prediction_id"] = [prediction_id(row) for row in output.itertuples()]
    output["churn_probability"] = probability
    output["predicted_churn"] = (probability >= THRESHOLD).astype(int)
    output["model_version"] = MODEL_VERSION
    output["schema_version"] = SCHEMA_VERSION
    output["batch_id"] = BATCH_ID
    output["run_id"] = RUN_ID
    output["scored_at_utc"] = now()
    assert len(output) == len(accepted)
    assert output["prediction_id"].is_unique
    assert output["churn_probability"].between(0, 1).all()
    prediction_path = TABLE_DIR / "09-batch-predictions.csv"
    output.to_csv(prediction_path, index=False, date_format="%Y-%m-%d")
    log_event(events, "predictions_verified", prediction_rows=len(output), predicted_positive_rows=int(output["predicted_churn"].sum()))

    rule_columns = [column for column in validation.columns if column not in ["source_row", "accepted", "failure_reasons"]]
    rule_counts = validation[rule_columns].sum().astype(int)
    summary = pd.DataFrame([
        {"metric": "input_rows", "value": len(batch)}, {"metric": "accepted_rows", "value": len(accepted)},
        {"metric": "rejected_rows", "value": len(rejected)}, {"metric": "rejection_rate", "value": len(rejected) / len(batch)},
        {"metric": "predicted_positive_rate", "value": output["predicted_churn"].mean()},
        {"metric": "mean_churn_probability", "value": output["churn_probability"].mean()},
    ])
    summary.to_csv(TABLE_DIR / "09-run-summary.csv", index=False)
    completed_at = now()
    manifest = {
        "run_id": RUN_ID, "batch_id": BATCH_ID, "status": "succeeded_with_rejections" if len(rejected) else "succeeded",
        "model_version": MODEL_VERSION, "schema_version": SCHEMA_VERSION, "decision_threshold": THRESHOLD,
        "started_at_utc": events[0]["timestamp_utc"], "completed_at_utc": completed_at,
        "duration_seconds": round(time.perf_counter() - started, 4), "input_rows": len(batch),
        "accepted_rows": len(accepted), "rejected_rows": len(rejected), "prediction_rows": len(output),
        "input_file": str(batch_path.relative_to(ROOT)), "input_sha256": digest(batch_path),
        "prediction_file": str(prediction_path.relative_to(ROOT)), "prediction_sha256": digest(prediction_path),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log_event(events, "run_completed", status=manifest["status"], duration_seconds=manifest["duration_seconds"])
    LOG_PATH.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    axes[0, 0].bar(["Accepted", "Rejected"], [len(accepted), len(rejected)], color=["#59A14F", "#E15759"])
    axes[0, 0].set_title("Input-row disposition")
    axes[0, 0].set_ylabel("Rows")
    axes[0, 1].barh(rule_counts.sort_values().index, rule_counts.sort_values().values, color="#E15759")
    axes[0, 1].set_title("Validation-rule failures")
    axes[0, 1].set_xlabel("Rows")
    axes[1, 0].hist(output["churn_probability"], bins=20, color="#4C78A8", edgecolor="white")
    axes[1, 0].axvline(THRESHOLD, linestyle="--", color="#333333", label="Decision threshold")
    axes[1, 0].set_title("Accepted prediction distribution")
    axes[1, 0].set_xlabel("Churn probability")
    axes[1, 0].legend(frameon=False)
    plan_risk = output.groupby("plan")["churn_probability"].mean().sort_values()
    axes[1, 1].bar(plan_risk.index, plan_risk.values, color="#F28E2B")
    axes[1, 1].set_title("Mean predicted risk by plan")
    axes[1, 1].set_ylabel("Mean probability")
    fig.suptitle("Operational validation makes batch predictions trustworthy", fontsize=16, fontweight="bold")
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {prediction_path.relative_to(ROOT)}")
    print(f"Saved {(ROOT / 'data' / 'quarantine' / '09-rejected-rows.csv').relative_to(ROOT)}")
    print(f"Saved {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"Saved {LOG_PATH.relative_to(ROOT)}")
    print(f"Saved {FIGURE_PATH.relative_to(ROOT)}")
    print(f"Accepted {len(accepted)} rows; rejected {len(rejected)} rows")


if __name__ == "__main__":
    main()
