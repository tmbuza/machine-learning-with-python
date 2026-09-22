"""Train baseline and first supervised models for MLPY-004."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "04-customer-modelling-table.csv"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_PATH = ROOT / "results" / "figures" / "04-baselines-and-first-models.png"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charge",
    "support_contacts",
    "current_usage_hours",
]
CATEGORICAL_FEATURES = ["plan", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def create_modelling_table() -> pd.DataFrame:
    """Create repeated monthly customer records for two supervised tasks."""
    rng = np.random.default_rng(SEED)
    customer_count = 240
    dates = pd.date_range("2026-01-31", periods=4, freq="ME")
    customer_ids = np.array([f"C{value:04d}" for value in range(1, customer_count + 1)])
    plans = rng.choice(["Basic", "Standard", "Premium"], customer_count, p=[0.42, 0.38, 0.20])
    regions = rng.choice(["North", "South", "East", "West"], customer_count)
    customer_risk = rng.normal(0, 0.75, customer_count)
    base_usage = rng.normal(42, 10, customer_count)

    rows = []
    for month, snapshot_date in enumerate(dates):
        for position, customer_id in enumerate(customer_ids):
            tenure = int(rng.integers(2, 65) + month)
            contacts = int(rng.poisson(1.1 + max(customer_risk[position], 0) * 0.5))
            plan_charge = {"Basic": 35, "Standard": 65, "Premium": 95}[plans[position]]
            charge = float(plan_charge + rng.normal(0, 5))
            current_usage = float(
                base_usage[position]
                + {"Basic": -7, "Standard": 2, "Premium": 9}[plans[position]]
                + 1.3 * month
                + rng.normal(0, 5)
            )
            next_usage = float(
                5
                + 0.78 * current_usage
                - 1.7 * contacts
                + (4 if plans[position] == "Premium" else 0)
                + rng.normal(0, 5)
            )
            churn_logit = (
                -1.5
                + customer_risk[position]
                + 0.5 * contacts
                - 0.018 * tenure
                - 0.04 * current_usage
                + 0.18 * month
                + (0.35 if plans[position] == "Basic" else 0)
            )
            churn_probability = 1 / (1 + np.exp(-churn_logit))
            rows.append(
                {
                    "customer_id": customer_id,
                    "snapshot_date": snapshot_date,
                    "plan": plans[position],
                    "region": regions[position],
                    "tenure_months": tenure,
                    "monthly_charge": round(charge, 2),
                    "support_contacts": contacts,
                    "current_usage_hours": round(max(current_usage, 0), 2),
                    "churned_next_month": int(rng.random() < churn_probability),
                    "next_month_usage_hours": round(max(next_usage, 0), 2),
                }
            )

    data = pd.DataFrame(rows)
    missing_charge = rng.choice(data.index, size=24, replace=False)
    missing_plan = rng.choice(data.index, size=15, replace=False)
    data.loc[missing_charge, "monthly_charge"] = np.nan
    data.loc[missing_plan, "plan"] = np.nan
    return data


def make_preprocessor() -> ColumnTransformer:
    """Return a fresh mixed-data preprocessor for one pipeline."""
    numeric = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def fit_classifiers(X_train, y_train, X_test, y_test):
    """Fit and evaluate baseline, logistic, and shallow-tree classifiers."""
    candidates = {
        "Prior baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        ),
        "Shallow tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=4, min_samples_leaf=20, random_state=SEED
        ),
    }
    rows = []
    predictions = {}
    probabilities = {}
    for name, estimator in candidates.items():
        workflow = make_pipeline(make_preprocessor(), estimator)
        workflow.fit(X_train, y_train)
        predicted = workflow.predict(X_test)
        probability = workflow.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predicted),
                "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
                "roc_auc": roc_auc_score(y_test, probability),
            }
        )
        predictions[name] = predicted
        probabilities[name] = probability
    return pd.DataFrame(rows), predictions, probabilities


def fit_regressors(X_train, y_train, X_test, y_test):
    """Fit and evaluate mean, linear, and shallow-tree regressors."""
    candidates = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear regression": LinearRegression(),
        "Shallow tree": DecisionTreeRegressor(
            max_depth=4, min_samples_leaf=20, random_state=SEED
        ),
    }
    rows = []
    predictions = {}
    for name, estimator in candidates.items():
        workflow = make_pipeline(make_preprocessor(), estimator)
        workflow.fit(X_train, y_train)
        predicted = workflow.predict(X_test)
        rows.append(
            {
                "model": name,
                "mae": mean_absolute_error(y_test, predicted),
                "rmse": mean_squared_error(y_test, predicted) ** 0.5,
                "r_squared": r2_score(y_test, predicted),
            }
        )
        predictions[name] = predicted
    return pd.DataFrame(rows), predictions


def make_figure(
    classification_results,
    classification_predictions,
    regression_results,
    regression_predictions,
    classification_target,
    regression_target,
) -> None:
    """Create a four-panel first-model comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    blue = "#4C78A8"
    orange = "#F28E2B"

    x = np.arange(len(classification_results))
    width = 0.34
    axes[0, 0].bar(
        x - width / 2,
        classification_results["balanced_accuracy"],
        width,
        label="Balanced accuracy",
        color=blue,
    )
    axes[0, 0].bar(
        x + width / 2,
        classification_results["roc_auc"],
        width,
        label="ROC AUC",
        color=orange,
    )
    axes[0, 0].set_title("Classification performance")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_xticks(x, ["Baseline", "Logistic", "Tree"])
    axes[0, 0].legend(frameon=False)

    ConfusionMatrixDisplay.from_predictions(
        classification_target,
        classification_predictions["Logistic regression"],
        display_labels=["No churn", "Churn"],
        cmap="Blues",
        colorbar=False,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Logistic regression: held-out cases")

    regression_order = ["Mean baseline", "Linear regression", "Shallow tree"]
    regression_mae = regression_results.set_index("model").loc[regression_order, "mae"]
    axes[1, 0].bar(["Baseline", "Linear", "Tree"], regression_mae, color=["#9C9C9C", blue, orange])
    axes[1, 0].set_title("Regression error")
    axes[1, 0].set_ylabel("Mean absolute error (usage hours)")

    predicted_usage = regression_predictions["Linear regression"]
    axes[1, 1].scatter(regression_target, predicted_usage, alpha=0.55, color=blue, edgecolor="none")
    limits = [
        min(float(regression_target.min()), float(predicted_usage.min())),
        max(float(regression_target.max()), float(predicted_usage.max())),
    ]
    axes[1, 1].plot(limits, limits, linestyle="--", color="#333333", linewidth=1.5)
    axes[1, 1].set_xlim(limits)
    axes[1, 1].set_ylim(limits)
    axes[1, 1].set_title("Linear regression: observed versus predicted")
    axes[1, 1].set_xlabel("Observed usage hours")
    axes[1, 1].set_ylabel("Predicted usage hours")

    fig.suptitle("Baselines reveal whether first models add value", fontsize=16, fontweight="bold")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate data, train candidates, and save auditable results."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    data = create_modelling_table()
    data.to_csv(DATA_PATH, index=False, date_format="%Y-%m-%d")
    cutoff = data["snapshot_date"].max()
    train = data[data["snapshot_date"] < cutoff].copy()
    test = data[data["snapshot_date"] >= cutoff].copy()
    assert train["snapshot_date"].max() < test["snapshot_date"].min()

    X_train = train[FEATURES]
    X_test = test[FEATURES]
    classification_results, class_predictions, class_probabilities = fit_classifiers(
        X_train,
        train["churned_next_month"],
        X_test,
        test["churned_next_month"],
    )
    regression_results, regression_predictions = fit_regressors(
        X_train,
        train["next_month_usage_hours"],
        X_test,
        test["next_month_usage_hours"],
    )

    classification_results.round(4).to_csv(
        TABLE_DIR / "04-classification-results.csv", index=False
    )
    regression_results.round(4).to_csv(
        TABLE_DIR / "04-regression-results.csv", index=False
    )

    prediction_table = test[
        ["customer_id", "snapshot_date", "plan", "region", "churned_next_month", "next_month_usage_hours"]
    ].copy()
    prediction_table["logistic_churn_probability"] = class_probabilities["Logistic regression"]
    prediction_table["logistic_churn_prediction"] = class_predictions["Logistic regression"]
    prediction_table["linear_usage_prediction"] = regression_predictions["Linear regression"]
    prediction_table.to_csv(
        TABLE_DIR / "04-test-predictions.csv", index=False, date_format="%Y-%m-%d"
    )

    make_figure(
        classification_results,
        class_predictions,
        regression_results,
        regression_predictions,
        test["churned_next_month"],
        test["next_month_usage_hours"],
    )

    print(f"Saved {DATA_PATH.relative_to(ROOT)}")
    print(f"Saved {(TABLE_DIR / '04-classification-results.csv').relative_to(ROOT)}")
    print(f"Saved {(TABLE_DIR / '04-regression-results.csv').relative_to(ROOT)}")
    print(f"Saved {(TABLE_DIR / '04-test-predictions.csv').relative_to(ROOT)}")
    print(f"Saved {FIGURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
