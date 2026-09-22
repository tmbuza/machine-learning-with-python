"""Generate drift, performance, and retraining evidence for MLPY-010."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = ROOT / "data" / "reference" / "10-monitoring-reference.csv"
MONITORING_PATH = ROOT / "data" / "monitoring" / "10-monthly-monitoring-data.csv"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_PATH = ROOT / "results" / "figures" / "10-drift-and-performance-monitoring.png"
NUMERIC = ["tenure_months", "monthly_charge", "support_contacts", "usage_hours"]
CATEGORICAL = ["plan", "region"]
FEATURES = NUMERIC + CATEGORICAL


def generate_data():
    rng = np.random.default_rng(SEED)
    rows = []
    dates = pd.date_range("2026-01-31", periods=10, freq="ME")
    for month, date in enumerate(dates):
        count = 520
        drift = max(month - 3, 0)
        plans = rng.choice(["Basic", "Standard", "Premium"], count, p=np.array([0.42 + 0.025 * drift, 0.38 - 0.015 * drift, 0.20 - 0.01 * drift]))
        regions = rng.choice(["North", "South", "East", "West"], count)
        risk = rng.normal(0.08 * drift, 0.75, count)
        tenure = rng.integers(2, 70, count) + month
        contacts = rng.poisson(1.1 + 0.12 * drift + 0.35 * np.maximum(risk, 0))
        usage = rng.normal(43 - 1.3 * drift, 10, count) + np.where(plans == "Premium", 8, np.where(plans == "Basic", -6, 1))
        charge = np.select([plans == "Basic", plans == "Standard"], [35, 65], default=95) + rng.normal(0, 5, count)
        relationship_shift = np.where(month >= 7, 0.55 * (regions == "South") - 0.35 * contacts, 0)
        logit = -1.45 + risk + 0.5 * contacts - 0.018 * tenure - 0.04 * usage + 0.32 * (plans == "Basic") + relationship_shift
        probability = 1 / (1 + np.exp(-logit))
        target = (rng.random(count) < probability).astype(float)
        if month >= 8:
            target[:] = np.nan
        month_frame = pd.DataFrame({"snapshot_date": date, "tenure_months": tenure, "monthly_charge": charge.round(2), "support_contacts": contacts, "usage_hours": usage.round(2), "plan": plans, "region": regions, "churned_next_month": target})
        if month >= 5:
            missing = rng.choice(month_frame.index, size=10 + 4 * drift, replace=False)
            month_frame.loc[missing, "monthly_charge"] = np.nan
        rows.append(month_frame)
    data = pd.concat(rows, ignore_index=True)
    reference = data[data["snapshot_date"].isin(dates[:3])].copy()
    monitoring = data[data["snapshot_date"].isin(dates[3:])].copy()
    return reference, monitoring


def pipeline():
    prep = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
    ])
    return make_pipeline(prep, LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))


def proportions(values, edges):
    clean = pd.Series(values).dropna().to_numpy()
    counts, _ = np.histogram(clean, bins=edges)
    result = counts / max(counts.sum(), 1)
    return np.clip(result, 1e-6, None)


def psi(reference, current):
    edges = np.unique(np.quantile(pd.Series(reference).dropna(), np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    p, q = proportions(reference, edges), proportions(current, edges)
    return float(np.sum((q - p) * np.log(q / p)))


def categorical_distance(reference, current):
    categories = sorted(set(pd.Series(reference).fillna("Missing")) | set(pd.Series(current).fillna("Missing")))
    p = pd.Series(reference).fillna("Missing").value_counts(normalize=True).reindex(categories, fill_value=0)
    q = pd.Series(current).fillna("Missing").value_counts(normalize=True).reindex(categories, fill_value=0)
    return float(0.5 * np.abs(q - p).sum())


def severity(value, warning, critical):
    return "critical" if value >= critical else "warning" if value >= warning else "normal"


def main():
    for path in [REFERENCE_PATH.parent, MONITORING_PATH.parent, TABLE_DIR, FIGURE_PATH.parent]:
        path.mkdir(parents=True, exist_ok=True)
    reference, monitoring = generate_data()
    reference.to_csv(REFERENCE_PATH, index=False, date_format="%Y-%m-%d")
    monitoring.to_csv(MONITORING_PATH, index=False, date_format="%Y-%m-%d")
    model = pipeline()
    model.fit(reference[FEATURES], reference["churned_next_month"].astype(int))
    reference_probability = model.predict_proba(reference[FEATURES])[:, 1]
    monitoring["churn_probability"] = model.predict_proba(monitoring[FEATURES])[:, 1]
    reference_auc = roc_auc_score(reference["churned_next_month"], reference_probability)

    drift_rows, prediction_rows, performance_rows, alert_rows = [], [], [], []
    for month, frame in monitoring.groupby("snapshot_date"):
        label = month.strftime("%Y-%m")
        for feature in NUMERIC:
            value = psi(reference[feature], frame[feature])
            level = severity(value, 0.10, 0.25)
            drift_rows.append({"month": label, "feature": feature, "metric": "psi", "value": value, "severity": level, "reference_missing_rate": reference[feature].isna().mean(), "current_missing_rate": frame[feature].isna().mean()})
            if level != "normal": alert_rows.append({"month": label, "alert_type": "feature_drift", "item": feature, "value": value, "severity": level})
        for feature in CATEGORICAL:
            value = categorical_distance(reference[feature], frame[feature])
            level = severity(value, 0.10, 0.20)
            drift_rows.append({"month": label, "feature": feature, "metric": "total_variation_distance", "value": value, "severity": level, "reference_missing_rate": reference[feature].isna().mean(), "current_missing_rate": frame[feature].isna().mean()})
            if level != "normal": alert_rows.append({"month": label, "alert_type": "feature_drift", "item": feature, "value": value, "severity": level})
        prediction_psi = psi(reference_probability, frame["churn_probability"])
        prediction_rows.append({"month": label, "rows": len(frame), "mean_probability": frame["churn_probability"].mean(), "predicted_positive_rate": (frame["churn_probability"] >= 0.55).mean(), "prediction_psi": prediction_psi, "severity": severity(prediction_psi, 0.10, 0.25)})
        labelled = frame["churned_next_month"].notna()
        if labelled.all():
            auc = roc_auc_score(frame["churned_next_month"], frame["churn_probability"])
            brier = brier_score_loss(frame["churned_next_month"], frame["churn_probability"])
            performance_rows.append({"month": label, "label_status": "mature", "roc_auc": auc, "brier_score": brier, "auc_change_from_reference": auc - reference_auc})
            if auc < reference_auc - 0.08:
                alert_rows.append({"month": label, "alert_type": "performance_drop", "item": "roc_auc", "value": auc, "severity": "critical"})
        else:
            performance_rows.append({"month": label, "label_status": "unavailable", "roc_auc": np.nan, "brier_score": np.nan, "auc_change_from_reference": np.nan})

    drift = pd.DataFrame(drift_rows)
    predictions = pd.DataFrame(prediction_rows)
    performance = pd.DataFrame(performance_rows)
    alerts = pd.DataFrame(alert_rows, columns=["month", "alert_type", "item", "value", "severity"])
    recent_drift = drift[drift["month"].isin(sorted(drift["month"].unique())[-3:])]
    mature_perf = performance[performance["label_status"] == "mature"]
    has_drift = (recent_drift["severity"].isin(["warning", "critical"])).any()
    has_drop = (mature_perf["auc_change_from_reference"] < -0.08).any()
    recommendation = "prepare_and_validate_challenger" if has_drift and has_drop else "investigate_and_continue_monitoring" if has_drift else "continue_monitoring"
    assessment = pd.DataFrame([{"reference_roc_auc": reference_auc, "recent_drift_detected": has_drift, "mature_performance_drop_detected": has_drop, "latest_labels_available": performance.iloc[-1]["label_status"] == "mature", "recommended_action": recommendation, "automatic_retraining": False}])
    drift.round(5).to_csv(TABLE_DIR / "10-feature-drift.csv", index=False)
    predictions.round(5).to_csv(TABLE_DIR / "10-prediction-monitoring.csv", index=False)
    performance.round(5).to_csv(TABLE_DIR / "10-performance-monitoring.csv", index=False)
    alerts.round(5).to_csv(TABLE_DIR / "10-monitoring-alerts.csv", index=False)
    assessment.round(5).to_csv(TABLE_DIR / "10-retraining-assessment.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    matrix = drift.pivot(index="feature", columns="month", values="value").reindex(FEATURES)
    image = axes[0, 0].imshow(matrix.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(0.3, np.nanmax(matrix.values)))
    axes[0, 0].set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    axes[0, 0].set_yticks(range(len(matrix.index)), matrix.index)
    axes[0, 0].set_title("Feature drift metrics")
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.04)
    axes[0, 1].plot(predictions["month"], predictions["prediction_psi"], marker="o", label="Prediction PSI", color="#E15759")
    axes[0, 1].plot(predictions["month"], predictions["mean_probability"], marker="o", label="Mean probability", color="#4C78A8")
    axes[0, 1].axhline(0.10, linestyle="--", color="#777777")
    axes[0, 1].tick_params(axis="x", rotation=35)
    axes[0, 1].set_title("Prediction monitoring")
    axes[0, 1].legend(frameon=False)
    mature = performance[performance["label_status"] == "mature"]
    axes[1, 0].plot(mature["month"], mature["roc_auc"], marker="o", label="ROC AUC", color="#4C78A8")
    axes[1, 0].plot(mature["month"], mature["brier_score"], marker="o", label="Brier score", color="#F28E2B")
    axes[1, 0].axhline(reference_auc, linestyle="--", color="#4C78A8", alpha=0.6, label="Reference AUC")
    axes[1, 0].tick_params(axis="x", rotation=35)
    axes[1, 0].set_title("Performance after labels mature")
    axes[1, 0].legend(frameon=False)
    if alerts.empty:
        alert_counts = pd.DataFrame(0, index=predictions["month"], columns=["warning", "critical"])
    else:
        alert_counts = alerts.groupby(["month", "severity"]).size().unstack(fill_value=0).reindex(predictions["month"], fill_value=0)
        for column in ["warning", "critical"]:
            if column not in alert_counts: alert_counts[column] = 0
    axes[1, 1].bar(alert_counts.index, alert_counts["warning"], label="Warning", color="#F28E2B")
    axes[1, 1].bar(alert_counts.index, alert_counts["critical"], bottom=alert_counts["warning"], label="Critical", color="#E15759")
    axes[1, 1].tick_params(axis="x", rotation=35)
    axes[1, 1].set_title("Actionable monitoring alerts")
    axes[1, 1].set_ylabel("Alerts")
    axes[1, 1].legend(frameon=False)
    fig.suptitle("Drift is investigated with performance and operational context", fontsize=16, fontweight="bold")
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    for name in ["10-feature-drift.csv", "10-prediction-monitoring.csv", "10-performance-monitoring.csv", "10-monitoring-alerts.csv", "10-retraining-assessment.csv"]:
        print(f"Saved {(TABLE_DIR / name).relative_to(ROOT)}")
    print(f"Saved {FIGURE_PATH.relative_to(ROOT)}")
    print(f"Recommended action: {recommendation}")


if __name__ == "__main__":
    main()
