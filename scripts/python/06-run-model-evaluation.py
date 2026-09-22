"""Generate classification, regression, and subgroup diagnostics for MLPY-006."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "06-evaluation-data.csv"
TABLE_DIR = ROOT / "results" / "tables"
CLASS_FIGURE = ROOT / "results" / "figures" / "06-classification-diagnostics.png"
REG_FIGURE = ROOT / "results" / "figures" / "06-regression-and-subgroup-diagnostics.png"
NUMERIC = ["tenure_months", "monthly_charge", "support_contacts", "current_usage_hours"]
CATEGORICAL = ["plan", "region"]
FEATURES = NUMERIC + CATEGORICAL


def create_data() -> pd.DataFrame:
    """Create seven monthly customer snapshots with two prediction targets."""
    rng = np.random.default_rng(SEED)
    count = 360
    dates = pd.date_range("2026-01-31", periods=7, freq="ME")
    identifiers = [f"C{number:04d}" for number in range(1, count + 1)]
    plans = rng.choice(["Basic", "Standard", "Premium"], count, p=[0.42, 0.38, 0.20])
    regions = rng.choice(["North", "South", "East", "West"], count)
    risk = rng.normal(0, 0.75, count)
    base_usage = rng.normal(43, 9, count)
    rows = []
    for month, date in enumerate(dates):
        for position, customer_id in enumerate(identifiers):
            tenure = int(rng.integers(2, 70) + month)
            contacts = int(rng.poisson(1.1 + max(risk[position], 0) * 0.45))
            charge = {"Basic": 35, "Standard": 65, "Premium": 95}[plans[position]] + rng.normal(0, 5)
            usage = base_usage[position] + {"Basic": -7, "Standard": 1, "Premium": 9}[plans[position]] + month + rng.normal(0, 5)
            next_usage = 5 + 0.79 * usage - 1.6 * contacts + (4 if plans[position] == "Premium" else 0) + rng.normal(0, 5)
            logit = -1.45 + risk[position] + 0.5 * contacts - 0.018 * tenure - 0.038 * usage + 0.09 * month + (0.35 if plans[position] == "Basic" else 0)
            probability = 1 / (1 + np.exp(-logit))
            rows.append({
                "customer_id": customer_id, "snapshot_date": date, "plan": plans[position], "region": regions[position],
                "tenure_months": tenure, "monthly_charge": round(float(charge), 2), "support_contacts": contacts,
                "current_usage_hours": round(max(float(usage), 0), 2),
                "churned_next_month": int(rng.random() < probability),
                "next_month_usage_hours": round(max(float(next_usage), 0), 2),
            })
    data = pd.DataFrame(rows)
    data.loc[rng.choice(data.index, 48, replace=False), "monthly_charge"] = np.nan
    data.loc[rng.choice(data.index, 28, replace=False), "plan"] = np.nan
    return data


def preprocessor() -> ColumnTransformer:
    """Create fresh preprocessing for numeric and categorical predictors."""
    return ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
    ])


def threshold_table(target, probability) -> pd.DataFrame:
    """Evaluate candidate thresholds on validation predictions."""
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.02):
        prediction = (probability >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "precision": precision_score(target, prediction, zero_division=0),
            "recall": recall_score(target, prediction, zero_division=0),
            "f1": f1_score(target, prediction, zero_division=0),
            "predicted_positive_rate": prediction.mean(),
        })
    return pd.DataFrame(rows)


def subgroup_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate classification and regression diagnostics by plan."""
    rows = []
    for group, frame in predictions.groupby("plan", dropna=False):
        rows.append({
            "group_variable": "plan", "group": "Missing" if pd.isna(group) else group,
            "rows": len(frame), "positive_cases": int(frame["observed_churn"].sum()),
            "positive_rate": frame["observed_churn"].mean(),
            "precision": precision_score(frame["observed_churn"], frame["predicted_churn"], zero_division=0),
            "recall": recall_score(frame["observed_churn"], frame["predicted_churn"], zero_division=0),
            "regression_mae": mean_absolute_error(frame["observed_usage"], frame["predicted_usage"]),
        })
    return pd.DataFrame(rows)


def classification_figure(y_test, probability, prediction, thresholds, selected_threshold):
    """Create ranking, threshold, confusion-matrix, and calibration diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    false_positive, true_positive, _ = roc_curve(y_test, probability)
    precision, recall, _ = precision_recall_curve(y_test, probability)
    axes[0, 0].plot(false_positive, true_positive, color="#4C78A8", linewidth=2, label=f"ROC AUC = {roc_auc_score(y_test, probability):.3f}")
    axes[0, 0].plot(recall, precision, color="#F28E2B", linewidth=2, label=f"Average precision = {average_precision_score(y_test, probability):.3f}")
    axes[0, 0].plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    axes[0, 0].set_title("Ranking curves")
    axes[0, 0].set_xlabel("False-positive rate / Recall")
    axes[0, 0].set_ylabel("True-positive rate / Precision")
    axes[0, 0].legend(frameon=False)

    for metric, color in [("precision", "#59A14F"), ("recall", "#E15759"), ("f1", "#4C78A8")]:
        axes[0, 1].plot(thresholds["threshold"], thresholds[metric], label=metric.title(), color=color, linewidth=2)
    axes[0, 1].axvline(selected_threshold, linestyle="--", color="#222222", label=f"Selected = {selected_threshold:.2f}")
    axes[0, 1].set_title("Validation threshold trade-offs")
    axes[0, 1].set_xlabel("Threshold")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].legend(frameon=False)

    ConfusionMatrixDisplay.from_predictions(y_test, prediction, display_labels=["No churn", "Churn"], cmap="Blues", colorbar=False, ax=axes[1, 0])
    axes[1, 0].set_title("Final-test confusion matrix")

    observed, predicted = calibration_curve(y_test, probability, n_bins=8, strategy="quantile")
    axes[1, 1].plot(predicted, observed, marker="o", color="#4C78A8", linewidth=2)
    axes[1, 1].plot([0, 1], [0, 1], linestyle="--", color="#333333")
    axes[1, 1].set_title(f"Calibration (Brier = {brier_score_loss(y_test, probability):.3f})")
    axes[1, 1].set_xlabel("Mean predicted probability")
    axes[1, 1].set_ylabel("Observed positive rate")
    fig.suptitle("Classification performance depends on the evaluation question", fontsize=16, fontweight="bold")
    CLASS_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CLASS_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def regression_figure(predictions, subgroup):
    """Create regression residual and subgroup diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    observed = predictions["observed_usage"]
    predicted = predictions["predicted_usage"]
    residual = observed - predicted
    limits = [min(observed.min(), predicted.min()), max(observed.max(), predicted.max())]
    axes[0, 0].scatter(observed, predicted, alpha=0.5, color="#4C78A8", edgecolor="none")
    axes[0, 0].plot(limits, limits, linestyle="--", color="#333333")
    axes[0, 0].set_title("Observed versus predicted usage")
    axes[0, 0].set_xlabel("Observed hours")
    axes[0, 0].set_ylabel("Predicted hours")
    axes[0, 1].scatter(predicted, residual, alpha=0.5, color="#F28E2B", edgecolor="none")
    axes[0, 1].axhline(0, linestyle="--", color="#333333")
    axes[0, 1].set_title("Residuals versus predictions")
    axes[0, 1].set_xlabel("Predicted hours")
    axes[0, 1].set_ylabel("Observed − predicted")
    axes[1, 0].hist(residual, bins=24, color="#4C78A8", edgecolor="white")
    axes[1, 0].axvline(0, linestyle="--", color="#333333")
    axes[1, 0].set_title("Residual distribution")
    axes[1, 0].set_xlabel("Residual (hours)")
    shown = subgroup[subgroup["group"] != "Missing"].sort_values("group")
    axes[1, 1].bar(shown["group"], shown["recall"], color="#59A14F")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Churn recall by plan")
    axes[1, 1].set_ylabel("Recall")
    for position, row in enumerate(shown.itertuples()):
        axes[1, 1].text(position, row.recall + 0.03, f"n={row.rows}", ha="center")
    fig.suptitle("Inspect error structure and who experiences it", fontsize=16, fontweight="bold")
    REG_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(REG_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Train, select a threshold, evaluate once, and save diagnostics."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    data = create_data()
    data.to_csv(DATA_PATH, index=False, date_format="%Y-%m-%d")
    months = sorted(data["snapshot_date"].unique())
    train = data[data["snapshot_date"].isin(months[:-2])]
    validation = data[data["snapshot_date"] == months[-2]]
    development = data[data["snapshot_date"].isin(months[:-1])]
    test = data[data["snapshot_date"] == months[-1]].copy()

    classifier = make_pipeline(preprocessor(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))
    classifier.fit(train[FEATURES], train["churned_next_month"])
    validation_probability = classifier.predict_proba(validation[FEATURES])[:, 1]
    thresholds = threshold_table(validation["churned_next_month"], validation_probability)
    selected_threshold = float(thresholds.loc[thresholds["f1"].idxmax(), "threshold"])
    classifier.fit(development[FEATURES], development["churned_next_month"])
    test_probability = classifier.predict_proba(test[FEATURES])[:, 1]
    test_prediction = (test_probability >= selected_threshold).astype(int)

    regressor = make_pipeline(preprocessor(), LinearRegression())
    regressor.fit(development[FEATURES], development["next_month_usage_hours"])
    usage_prediction = regressor.predict(test[FEATURES])

    class_metrics = pd.DataFrame([{
        "selected_threshold": selected_threshold,
        "accuracy": accuracy_score(test["churned_next_month"], test_prediction),
        "balanced_accuracy": balanced_accuracy_score(test["churned_next_month"], test_prediction),
        "precision": precision_score(test["churned_next_month"], test_prediction, zero_division=0),
        "recall": recall_score(test["churned_next_month"], test_prediction, zero_division=0),
        "f1": f1_score(test["churned_next_month"], test_prediction, zero_division=0),
        "roc_auc": roc_auc_score(test["churned_next_month"], test_probability),
        "average_precision": average_precision_score(test["churned_next_month"], test_probability),
        "brier_score": brier_score_loss(test["churned_next_month"], test_probability),
        "log_loss": log_loss(test["churned_next_month"], test_probability),
    }])
    reg_metrics = pd.DataFrame([{
        "mae": mean_absolute_error(test["next_month_usage_hours"], usage_prediction),
        "rmse": mean_squared_error(test["next_month_usage_hours"], usage_prediction) ** 0.5,
        "r_squared": r2_score(test["next_month_usage_hours"], usage_prediction),
    }])
    predictions = test[["customer_id", "snapshot_date", "plan", "region"]].copy()
    predictions["observed_churn"] = test["churned_next_month"].to_numpy()
    predictions["churn_probability"] = test_probability
    predictions["selected_threshold"] = selected_threshold
    predictions["predicted_churn"] = test_prediction
    predictions["observed_usage"] = test["next_month_usage_hours"].to_numpy()
    predictions["predicted_usage"] = usage_prediction
    predictions["usage_residual"] = predictions["observed_usage"] - predictions["predicted_usage"]
    subgroup = subgroup_table(predictions)

    class_metrics.round(4).to_csv(TABLE_DIR / "06-classification-metrics.csv", index=False)
    thresholds.round(4).to_csv(TABLE_DIR / "06-threshold-analysis.csv", index=False)
    reg_metrics.round(4).to_csv(TABLE_DIR / "06-regression-metrics.csv", index=False)
    subgroup.round(4).to_csv(TABLE_DIR / "06-subgroup-performance.csv", index=False)
    predictions.round(4).to_csv(TABLE_DIR / "06-test-predictions.csv", index=False, date_format="%Y-%m-%d")
    classification_figure(test["churned_next_month"], test_probability, test_prediction, thresholds, selected_threshold)
    regression_figure(predictions, subgroup)

    print(f"Saved {DATA_PATH.relative_to(ROOT)}")
    for name in ["06-classification-metrics.csv", "06-threshold-analysis.csv", "06-regression-metrics.csv", "06-subgroup-performance.csv", "06-test-predictions.csv"]:
        print(f"Saved {(TABLE_DIR / name).relative_to(ROOT)}")
    print(f"Saved {CLASS_FIGURE.relative_to(ROOT)}")
    print(f"Saved {REG_FIGURE.relative_to(ROOT)}")
    print(f"Validation-selected threshold: {selected_threshold:.2f}")


if __name__ == "__main__":
    main()
