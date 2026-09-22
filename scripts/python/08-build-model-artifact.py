"""Train and persist a governed churn-prediction artifact for MLPY-008."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform

from joblib import dump
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
TABLE_DIR = ROOT / "results" / "tables"
ARTIFACT_PATH = ARTIFACT_DIR / "08-churn-pipeline.joblib"
SCHEMA_PATH = ARTIFACT_DIR / "08-feature-schema.json"
METADATA_PATH = ARTIFACT_DIR / "08-model-metadata.json"
FEATURES = ["tenure_months", "monthly_charge", "support_contacts", "usage_hours", "plan", "region"]
NUMERIC = FEATURES[:4]
CATEGORICAL = FEATURES[4:]


def create_data():
    """Create training data and a future scoring batch."""
    rng = np.random.default_rng(SEED)
    rows = []
    count = 420
    plans = rng.choice(["Basic", "Standard", "Premium"], count, p=[0.42, 0.38, 0.20])
    regions = rng.choice(["North", "South", "East", "West"], count)
    risk = rng.normal(0, 0.75, count)
    base_usage = rng.normal(43, 9, count)
    ids = [f"C{number:04d}" for number in range(1, count + 1)]
    dates = pd.date_range("2026-01-31", periods=6, freq="ME")
    for month, date in enumerate(dates):
        for position, customer_id in enumerate(ids):
            tenure = int(rng.integers(2, 70) + month)
            contacts = int(rng.poisson(1.1 + 0.45 * max(risk[position], 0)))
            charge = {"Basic": 35, "Standard": 65, "Premium": 95}[plans[position]] + rng.normal(0, 5)
            usage = base_usage[position] + {"Basic": -7, "Standard": 1, "Premium": 9}[plans[position]] + month + rng.normal(0, 5)
            logit = -1.4 + risk[position] + 0.55 * contacts - 0.02 * tenure - 0.043 * usage + 0.08 * month + (0.38 if plans[position] == "Basic" else 0)
            probability = 1 / (1 + np.exp(-logit))
            rows.append({"customer_id": customer_id, "snapshot_date": date, "tenure_months": tenure, "monthly_charge": round(float(charge), 2), "support_contacts": contacts, "usage_hours": round(max(float(usage), 0), 2), "plan": plans[position], "region": regions[position], "churned_next_month": int(rng.random() < probability)})
    data = pd.DataFrame(rows)
    data.loc[rng.choice(data.index, 45, replace=False), "monthly_charge"] = np.nan
    development = data[data["snapshot_date"] < dates[-1]].copy()
    test = data[data["snapshot_date"] == dates[-1]].copy()
    scoring = test.sample(160, random_state=SEED).drop(columns="churned_next_month").rename(columns={"snapshot_date": "prediction_date"}).reset_index(drop=True)
    scoring.loc[:7, "monthly_charge"] = np.nan
    return development, test, scoring


def build_pipeline():
    """Create the complete raw-data-to-probability pipeline."""
    preprocessing = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
    ])
    return make_pipeline(preprocessing, LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    """Build the artifact, schema, metadata, and reference prediction set."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "scoring").mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    development, test, scoring = create_data()
    development.to_csv(DATA_DIR / "processed" / "08-training-data.csv", index=False, date_format="%Y-%m-%d")
    scoring.to_csv(DATA_DIR / "scoring" / "08-scoring-batch.csv", index=False, date_format="%Y-%m-%d")

    pipeline = build_pipeline()
    pipeline.fit(development[FEATURES], development["churned_next_month"])
    test_probability = pipeline.predict_proba(test[FEATURES])[:, 1]
    threshold = 0.55
    test_prediction = (test_probability >= threshold).astype(int)
    reference_probability = pipeline.predict_proba(scoring[FEATURES])[:, 1]
    pd.DataFrame({"customer_id": scoring["customer_id"], "reference_probability": reference_probability}).to_csv(TABLE_DIR / "08-reference-probabilities.csv", index=False)
    dump(pipeline, ARTIFACT_PATH)
    artifact_digest = sha256(ARTIFACT_PATH.read_bytes()).hexdigest()

    schema = {
        "schema_version": "1.0.0", "required_features": FEATURES,
        "identifier_fields": ["customer_id", "prediction_date"],
        "numeric_features": NUMERIC, "categorical_features": CATEGORICAL,
        "allowed_categories": {"plan": ["Basic", "Standard", "Premium"], "region": ["North", "South", "East", "West"]},
        "missing_values_permitted": ["monthly_charge", "plan"],
        "target": "churned_next_month", "positive_class": 1,
    }
    metadata = {
        "artifact_id": "customer-churn-pipeline", "model_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(development), "test_rows": len(test),
        "training_period_start": development["snapshot_date"].min().date().isoformat(),
        "training_period_end": development["snapshot_date"].max().date().isoformat(),
        "test_period": test["snapshot_date"].min().date().isoformat(),
        "decision_threshold": threshold,
        "test_roc_auc": round(float(roc_auc_score(test["churned_next_month"], test_probability)), 6),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(test["churned_next_month"], test_prediction)), 6),
        "python_version": platform.python_version(), "scikit_learn_version": sklearn.__version__,
        "artifact_file": ARTIFACT_PATH.name, "artifact_sha256": artifact_digest,
        "schema_file": SCHEMA_PATH.name,
    }
    write_json(SCHEMA_PATH, schema)
    write_json(METADATA_PATH, metadata)
    print(f"Saved {ARTIFACT_PATH.relative_to(ROOT)}")
    print(f"Saved {SCHEMA_PATH.relative_to(ROOT)}")
    print(f"Saved {METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
