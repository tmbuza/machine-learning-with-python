"""Validate, load, score, and verify the MLPY-008 model artifact."""

from hashlib import sha256
import json
from pathlib import Path

from joblib import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_PATH = ROOT / "results" / "figures" / "08-model-artifact-verification.png"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    """Verify provenance metadata, validate inputs, and create batch predictions."""
    schema = read_json(ARTIFACT_DIR / "08-feature-schema.json")
    metadata = read_json(ARTIFACT_DIR / "08-model-metadata.json")
    artifact_path = ARTIFACT_DIR / metadata["artifact_file"]
    scoring = pd.read_csv(ROOT / "data" / "scoring" / "08-scoring-batch.csv", parse_dates=["prediction_date"])
    reference = pd.read_csv(TABLE_DIR / "08-reference-probabilities.csv")
    required = schema["required_features"]
    missing_columns = sorted(set(required) - set(scoring.columns))
    if missing_columns:
        raise ValueError(f"Missing required features: {missing_columns}")
    digest = sha256(artifact_path.read_bytes()).hexdigest()
    if digest != metadata["artifact_sha256"]:
        raise ValueError("Artifact checksum does not match metadata")

    pipeline = load(artifact_path)
    probability = pipeline.predict_proba(scoring[required])[:, 1]
    prediction = (probability >= metadata["decision_threshold"]).astype(int)
    maximum_difference = float(np.max(np.abs(reference["reference_probability"].to_numpy() - probability)))
    if maximum_difference >= 1e-12:
        raise AssertionError(f"Reloaded predictions differ by {maximum_difference}")

    output = scoring[["customer_id", "prediction_date", "plan", "region"]].copy()
    output["churn_probability"] = probability
    output["predicted_churn"] = prediction
    output["model_version"] = metadata["model_version"]
    output["artifact_sha256"] = digest
    output.to_csv(TABLE_DIR / "08-batch-predictions.csv", index=False, date_format="%Y-%m-%d")
    verification = pd.DataFrame([{
        "artifact_checksum_verified": True, "required_columns_present": True,
        "prediction_rows": len(output), "maximum_reference_difference": maximum_difference,
        "prediction_equivalence_verified": maximum_difference < 1e-12,
    }])
    verification.to_csv(TABLE_DIR / "08-artifact-verification.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    axes[0, 0].scatter(reference["reference_probability"], probability, alpha=0.55, color="#4C78A8", edgecolor="none")
    axes[0, 0].plot([0, 1], [0, 1], linestyle="--", color="#333333")
    axes[0, 0].set_title("Predictions before and after serialization")
    axes[0, 0].set_xlabel("Reference probability")
    axes[0, 0].set_ylabel("Reloaded probability")
    axes[0, 1].hist(probability, bins=20, color="#F28E2B", edgecolor="white")
    axes[0, 1].axvline(metadata["decision_threshold"], linestyle="--", color="#333333", label="Decision threshold")
    axes[0, 1].set_title("Batch risk-score distribution")
    axes[0, 1].set_xlabel("Predicted churn probability")
    axes[0, 1].legend(frameon=False)
    missing = scoring[required].isna().sum().sort_values()
    axes[1, 0].barh(missing.index, missing.values, color="#59A14F")
    axes[1, 0].set_title("Missing values in scoring inputs")
    axes[1, 0].set_xlabel("Missing rows")
    plan_risk = output.groupby("plan", dropna=False)["churn_probability"].mean().sort_values()
    axes[1, 1].bar(plan_risk.index.astype(str), plan_risk.values, color="#4C78A8")
    axes[1, 1].set_ylim(0, max(0.65, plan_risk.max() + 0.05))
    axes[1, 1].set_title("Mean predicted risk by plan")
    axes[1, 1].set_ylabel("Mean probability")
    fig.suptitle("A model artifact is ready only after reproducibility checks", fontsize=16, fontweight="bold")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {(TABLE_DIR / '08-batch-predictions.csv').relative_to(ROOT)}")
    print(f"Saved {(TABLE_DIR / '08-artifact-verification.csv').relative_to(ROOT)}")
    print(f"Saved {FIGURE_PATH.relative_to(ROOT)}")
    print(f"Maximum prediction difference: {maximum_difference:.3g}")


if __name__ == "__main__":
    main()
