#!/usr/bin/env python3
"""Generate, structurally prepare, and profile Chapter 02 example data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 42
RAW_PATH = Path("data/raw/02-customer-retention-raw.csv")
PROCESSED_PATH = Path("data/processed/02-customer-retention-ml-ready.csv")
FIGURE_PATH = Path("results/figures/02-data-preparation-profile.png")
QUALITY_PATH = Path("results/tables/02-data-quality-summary.csv")
READINESS_PATH = Path("results/tables/02-feature-readiness-summary.csv")


def generate_raw_data() -> pd.DataFrame:
    """Create a reproducible customer snapshot table with known quality issues."""
    rng = np.random.default_rng(SEED)
    n = 420
    customer_id = np.array([f"CUST-{i:04d}" for i in range(1, n + 1)])
    snapshot_date = pd.Timestamp("2026-06-30")
    tenure = rng.integers(1, 97, size=n).astype(float)
    charges = np.clip(rng.normal(72, 24, size=n), 18, 180)
    support_calls = rng.poisson(1.7, size=n).astype(float)
    usage = np.clip(rng.normal(38, 16, size=n), 0, 110)
    region = rng.choice(["North", "South", "East", "West"], size=n, p=[0.24, 0.27, 0.26, 0.23])
    plan = rng.choice(
        ["Basic", "basic", "BASIC", "Standard", "standard ", "Premium", "premium", "Premium "],
        size=n,
        p=[0.18, 0.05, 0.03, 0.25, 0.06, 0.25, 0.11, 0.07],
    )
    contract = rng.choice(["Monthly", "Annual", "Two-year"], size=n, p=[0.56, 0.31, 0.13])

    logit = (
        -1.9
        - 0.018 * tenure
        + 0.021 * (charges - 70)
        + 0.34 * support_calls
        - 0.014 * usage
        + 0.7 * (contract == "Monthly")
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    frame = pd.DataFrame(
        {
            "customer_id": customer_id,
            "snapshot_date": snapshot_date,
            "tenure_months": tenure,
            "monthly_charges": charges.round(2),
            "support_calls_30d": support_calls,
            "usage_hours_30d": usage.round(1),
            "plan": plan,
            "contract_type": contract,
            "region": region,
            "churned_30d": churn,
        }
    )

    # Add post-outcome fields that must never enter the feature matrix.
    frame["cancellation_date"] = pd.NaT
    frame["cancellation_reason"] = pd.NA
    churned = frame["churned_30d"].eq(1)
    frame.loc[churned, "cancellation_date"] = snapshot_date + pd.to_timedelta(
        rng.integers(1, 31, size=churned.sum()), unit="D"
    )
    frame.loc[churned, "cancellation_reason"] = rng.choice(
        ["Price", "Service", "Moved", "Competitor"], size=churned.sum()
    )

    # Introduce missing and impossible values deliberately.
    frame.loc[rng.choice(n, 28, replace=False), "monthly_charges"] = np.nan
    frame.loc[rng.choice(n, 20, replace=False), "usage_hours_30d"] = np.nan
    frame.loc[rng.choice(n, 13, replace=False), "plan"] = pd.NA
    frame.loc[rng.choice(n, 8, replace=False), "tenure_months"] = -rng.integers(1, 10, size=8)
    frame.loc[rng.choice(n, 5, replace=False), "monthly_charges"] = -rng.integers(1, 50, size=5)

    # Duplicate confirmed snapshots exactly.
    duplicates = frame.sample(9, random_state=SEED)
    return pd.concat([frame, duplicates], ignore_index=True)


def prepare_structurally(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply fixed structural rules without fitting model preprocessing."""
    working = raw.copy()
    records: list[dict[str, object]] = []

    duplicate_count = int(working.duplicated().sum())
    working = working.drop_duplicates().copy()
    records.append(
        {
            "issue": "Exact duplicate snapshots",
            "count": duplicate_count,
            "action": "Removed confirmed exact duplicates",
            "stage": "Structural preparation",
        }
    )

    invalid_tenure = working["tenure_months"].lt(0)
    working.loc[invalid_tenure, "tenure_months"] = np.nan
    records.append(
        {
            "issue": "Negative tenure values",
            "count": int(invalid_tenure.sum()),
            "action": "Converted to missing; no imputation performed",
            "stage": "Structural preparation",
        }
    )

    invalid_charges = working["monthly_charges"].lt(0)
    working.loc[invalid_charges, "monthly_charges"] = np.nan
    records.append(
        {
            "issue": "Negative monthly charges",
            "count": int(invalid_charges.sum()),
            "action": "Converted to missing; no imputation performed",
            "stage": "Structural preparation",
        }
    )

    plan_map = {
        "basic": "Basic",
        "standard": "Standard",
        "premium": "Premium",
    }
    original_plan = working["plan"].copy()
    working["plan"] = working["plan"].astype("string").str.strip().str.lower().map(plan_map)
    standardized = original_plan.notna() & original_plan.fillna("<NA>").astype(str).ne(
        working["plan"].fillna("<NA>").astype(str)
    )
    records.append(
        {
            "issue": "Non-canonical plan labels",
            "count": int(standardized.sum()),
            "action": "Mapped documented variants to canonical labels",
            "stage": "Structural preparation",
        }
    )

    leakage_columns = ["cancellation_date", "cancellation_reason"]
    working = working.drop(columns=leakage_columns)
    records.append(
        {
            "issue": "Post-outcome leakage columns",
            "count": len(leakage_columns),
            "action": "Excluded from processed modelling table",
            "stage": "Prediction-time boundary",
        }
    )

    working["snapshot_date"] = pd.to_datetime(working["snapshot_date"], errors="raise")
    working = working.sort_values(["snapshot_date", "customer_id"]).reset_index(drop=True)
    return working, pd.DataFrame(records)


def build_readiness_summary(processed: pd.DataFrame) -> pd.DataFrame:
    """Describe the role and next modelling action for every processed column."""
    roles = {
        "customer_id": ("Identifier", "Retain for audit and grouped splitting; exclude from estimator"),
        "snapshot_date": ("Time", "Retain for temporal splitting and audit"),
        "tenure_months": ("Numeric feature", "Impute within training pipeline if required"),
        "monthly_charges": ("Numeric feature", "Impute and scale within training pipeline"),
        "support_calls_30d": ("Numeric feature", "Validate distribution; transform only inside pipeline"),
        "usage_hours_30d": ("Numeric feature", "Impute and scale within training pipeline"),
        "plan": ("Categorical feature", "Impute and encode within training pipeline"),
        "contract_type": ("Categorical feature", "Encode within training pipeline"),
        "region": ("Group / candidate feature", "Retain for subgroup evaluation; decide modelling role"),
        "churned_30d": ("Target", "Separate from feature matrix"),
    }
    rows = []
    for column in processed.columns:
        role, action = roles[column]
        rows.append(
            {
                "column": column,
                "role": role,
                "dtype": str(processed[column].dtype),
                "missing_count": int(processed[column].isna().sum()),
                "missing_percent": round(float(processed[column].isna().mean() * 100), 2),
                "unique_values": int(processed[column].nunique(dropna=True)),
                "next_action": action,
            }
        )
    return pd.DataFrame(rows)


def create_profile_figure(raw: pd.DataFrame, processed: pd.DataFrame) -> None:
    """Create a four-panel diagnostic profile for the chapter."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle("Data preparation profile", fontsize=18, fontweight="bold")

    missing = raw.isna().mean().mul(100).sort_values(ascending=False)
    missing = missing[missing.gt(0)]
    axes[0, 0].barh(missing.index[::-1], missing.values[::-1], color="#4E79A7")
    axes[0, 0].set(title="Missing values in raw data", xlabel="Missing (%)")

    axes[0, 1].hist(
        processed["tenure_months"].dropna(), bins=18, alpha=0.72, label="Tenure (months)", color="#59A14F"
    )
    axes[0, 1].hist(
        processed["monthly_charges"].dropna(), bins=18, alpha=0.62, label="Monthly charges", color="#F28E2B"
    )
    axes[0, 1].set(title="Numeric feature distributions", xlabel="Observed value", ylabel="Customers")
    axes[0, 1].legend(frameon=False)

    plan_counts = processed["plan"].value_counts().reindex(["Basic", "Standard", "Premium"])
    axes[1, 0].bar(plan_counts.index, plan_counts.values, color=["#76B7B2", "#4E79A7", "#B07AA1"])
    axes[1, 0].set(title="Canonical plan categories", xlabel="Plan", ylabel="Customers")

    churn_by_region = processed.groupby("region", observed=True)["churned_30d"].mean().mul(100).sort_values()
    axes[1, 1].barh(churn_by_region.index, churn_by_region.values, color="#E15759")
    axes[1, 1].set(title="Observed churn by region", xlabel="Churned within 30 days (%)")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.18, linewidth=0.7)

    fig.text(
        0.5,
        0.015,
        "Structural preparation preserves missing values for training-fitted pipelines.",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """Run the complete reproducible Chapter 02 workflow."""
    for path in [RAW_PATH, PROCESSED_PATH, FIGURE_PATH, QUALITY_PATH, READINESS_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)

    raw = generate_raw_data()
    raw.to_csv(RAW_PATH, index=False)

    processed, quality = prepare_structurally(raw)
    readiness = build_readiness_summary(processed)

    processed.to_csv(PROCESSED_PATH, index=False)
    quality.to_csv(QUALITY_PATH, index=False)
    readiness.to_csv(READINESS_PATH, index=False)
    create_profile_figure(raw, processed)

    print(f"Saved {RAW_PATH}")
    print(f"Saved {PROCESSED_PATH}")
    print(f"Saved {QUALITY_PATH}")
    print(f"Saved {READINESS_PATH}")
    print(f"Saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
