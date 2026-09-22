"""Generate global and case-level model interpretations for MLPY-007."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "07-interpretation-data.csv"
TABLE_DIR = ROOT / "results" / "tables"
GLOBAL_FIGURE = ROOT / "results" / "figures" / "07-global-model-interpretation.png"
CASE_FIGURE = ROOT / "results" / "figures" / "07-case-level-interpretation.png"
NUMERIC = ["tenure_months", "monthly_charge", "support_contacts", "usage_hours"]
CATEGORICAL = ["plan", "region"]
FEATURES = NUMERIC + CATEGORICAL


def create_data() -> pd.DataFrame:
    """Create repeated customer records with interpretable predictive structure."""
    rng = np.random.default_rng(SEED)
    count = 400
    dates = pd.date_range("2026-01-31", periods=6, freq="ME")
    ids = [f"C{number:04d}" for number in range(1, count + 1)]
    plans = rng.choice(["Basic", "Standard", "Premium"], count, p=[0.42, 0.38, 0.20])
    regions = rng.choice(["North", "South", "East", "West"], count)
    risk = rng.normal(0, 0.75, count)
    base_usage = rng.normal(43, 9, count)
    rows = []
    for month, date in enumerate(dates):
        for position, customer_id in enumerate(ids):
            tenure = int(rng.integers(2, 70) + month)
            contacts = int(rng.poisson(1.1 + 0.5 * max(risk[position], 0)))
            charge = {"Basic": 35, "Standard": 65, "Premium": 95}[plans[position]] + rng.normal(0, 5)
            usage = base_usage[position] + {"Basic": -7, "Standard": 1, "Premium": 9}[plans[position]] + month + rng.normal(0, 5)
            logit = -1.4 + risk[position] + 0.58 * contacts - 0.022 * tenure - 0.045 * usage + 0.08 * month + (0.4 if plans[position] == "Basic" else 0)
            probability = 1 / (1 + np.exp(-logit))
            rows.append({
                "customer_id": customer_id, "snapshot_date": date, "plan": plans[position], "region": regions[position],
                "tenure_months": tenure, "monthly_charge": round(float(charge), 2),
                "support_contacts": contacts, "usage_hours": round(max(float(usage), 0), 2),
                "churned_next_month": int(rng.random() < probability),
            })
    data = pd.DataFrame(rows)
    data.loc[rng.choice(data.index, 44, replace=False), "monthly_charge"] = np.nan
    data.loc[rng.choice(data.index, 24, replace=False), "plan"] = np.nan
    return data


def build_pipeline():
    """Create an interpretable mixed-data logistic pipeline."""
    preprocessing = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
    ])
    return make_pipeline(preprocessing, LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))


def coefficient_table(model) -> pd.DataFrame:
    """Extract transformed feature names and logistic coefficients."""
    names = model.named_steps["columntransformer"].get_feature_names_out()
    coefficients = model.named_steps["logisticregression"].coef_[0]
    table = pd.DataFrame({"transformed_feature": names, "coefficient": coefficients})
    table["odds_ratio"] = np.exp(table["coefficient"])
    table["absolute_coefficient"] = table["coefficient"].abs()
    return table.sort_values("absolute_coefficient", ascending=False)


def importance_table(model, X_test, y_test) -> pd.DataFrame:
    """Calculate repeated held-out permutation importance."""
    result = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED, n_jobs=-1)
    return pd.DataFrame({
        "feature": FEATURES,
        "mean_roc_auc_decrease": result.importances_mean,
        "sd_roc_auc_decrease": result.importances_std,
    }).sort_values("mean_roc_auc_decrease", ascending=False)


def response_curves(model, X_test) -> pd.DataFrame:
    """Create average prediction response curves for two numeric features."""
    rows = []
    for feature in ["usage_hours", "support_contacts"]:
        lower, upper = X_test[feature].quantile([0.05, 0.95])
        grid = np.unique(np.linspace(lower, upper, 24).round(2))
        for value in grid:
            modified = X_test.copy()
            modified[feature] = value
            rows.append({
                "feature": feature,
                "value": value,
                "mean_predicted_probability": model.predict_proba(modified)[:, 1].mean(),
            })
    return pd.DataFrame(rows)


def case_tables(model, development, test):
    """Select cases and calculate one-feature-at-a-time reference perturbations."""
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    quantiles = np.linspace(0.05, 0.95, 6)
    targets = np.quantile(probabilities, quantiles)
    chosen_positions = []
    for target in targets:
        order = np.argsort(np.abs(probabilities - target))
        chosen_positions.append(next(position for position in order if position not in chosen_positions))
    selected = test.iloc[chosen_positions].copy()
    selected["predicted_probability"] = probabilities[chosen_positions]
    selected["case_label"] = [f"Case {number}" for number in range(1, len(selected) + 1)]

    references = {feature: development[feature].median() for feature in NUMERIC}
    references.update({feature: development[feature].mode(dropna=True).iloc[0] for feature in CATEGORICAL})
    rows = []
    for _, case in selected.iterrows():
        original = float(case["predicted_probability"])
        for feature in FEATURES:
            modified = case[FEATURES].to_frame().T.copy()
            modified[feature] = references[feature]
            perturbed = float(model.predict_proba(modified)[:, 1][0])
            rows.append({
                "case_label": case["case_label"], "customer_id": case["customer_id"],
                "feature": feature, "observed_value": case[feature], "reference_value": references[feature],
                "original_probability": original, "reference_perturbed_probability": perturbed,
                "probability_difference": original - perturbed,
            })
    return selected, pd.DataFrame(rows)


def global_figure(coefficients, importance, responses):
    """Create coefficient, importance, and response-curve panels."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    shown = coefficients.head(10).sort_values("coefficient")
    colors = np.where(shown["coefficient"] >= 0, "#E15759", "#4C78A8")
    axes[0, 0].barh(shown["transformed_feature"].str.replace("numeric__", "").str.replace("categorical__", ""), shown["coefficient"], color=colors)
    axes[0, 0].axvline(0, color="#333333", linewidth=1)
    axes[0, 0].set_title("Largest standardized logistic coefficients")
    axes[0, 0].set_xlabel("Coefficient on log-odds scale")
    shown_importance = importance.sort_values("mean_roc_auc_decrease")
    axes[0, 1].barh(shown_importance["feature"], shown_importance["mean_roc_auc_decrease"], xerr=shown_importance["sd_roc_auc_decrease"], color="#59A14F", alpha=0.9)
    axes[0, 1].axvline(0, color="#333333", linewidth=1)
    axes[0, 1].set_title("Held-out permutation importance")
    axes[0, 1].set_xlabel("Mean decrease in ROC AUC")
    for axis, feature, title in [(axes[1, 0], "usage_hours", "Response to usage hours"), (axes[1, 1], "support_contacts", "Response to support contacts")]:
        frame = responses[responses["feature"] == feature]
        axis.plot(frame["value"], frame["mean_predicted_probability"], color="#F28E2B", linewidth=2.5)
        axis.set_title(title)
        axis.set_xlabel(feature.replace("_", " ").title())
        axis.set_ylabel("Mean predicted churn probability")
    fig.suptitle("Global methods describe different aspects of model behaviour", fontsize=16, fontweight="bold")
    GLOBAL_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(GLOBAL_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def case_figure(selected, perturbations):
    """Create selected-case score and perturbation panels."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    order = selected.sort_values("predicted_probability")
    axes[0].barh(order["case_label"], order["predicted_probability"], color="#4C78A8")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Predicted churn probability")
    axes[0].set_title("Selected cases across the risk range")
    matrix = perturbations.pivot(index="case_label", columns="feature", values="probability_difference")
    matrix = matrix.loc[[f"Case {number}" for number in range(1, 7)], FEATURES]
    limit = max(abs(matrix.values.min()), abs(matrix.values.max()))
    image = axes[1].imshow(matrix.values, cmap="RdBu_r", aspect="auto", vmin=-limit, vmax=limit)
    axes[1].set_xticks(range(len(matrix.columns)), [value.replace("_", " ") for value in matrix.columns], rotation=35, ha="right")
    axes[1].set_yticks(range(len(matrix.index)), matrix.index)
    axes[1].set_title("Probability change versus one-feature reference")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[1].text(column, row, f"{matrix.iloc[row, column]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="Original − perturbed probability")
    fig.suptitle("Case-level perturbations are comparisons, not causal effects", fontsize=16, fontweight="bold")
    CASE_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CASE_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Fit the pipeline and save global and case-level interpretation evidence."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    data = create_data()
    data.to_csv(DATA_PATH, index=False, date_format="%Y-%m-%d")
    final_month = data["snapshot_date"].max()
    development = data[data["snapshot_date"] < final_month].copy()
    test = data[data["snapshot_date"] == final_month].copy()
    model = build_pipeline()
    model.fit(development[FEATURES], development["churned_next_month"])
    test_auc = roc_auc_score(test["churned_next_month"], model.predict_proba(test[FEATURES])[:, 1])

    coefficients = coefficient_table(model)
    importance = importance_table(model, test[FEATURES], test["churned_next_month"])
    responses = response_curves(model, test[FEATURES])
    selected, perturbations = case_tables(model, development, test)
    coefficients.round(5).to_csv(TABLE_DIR / "07-logistic-coefficients.csv", index=False)
    importance.round(5).to_csv(TABLE_DIR / "07-permutation-importance.csv", index=False)
    responses.round(5).to_csv(TABLE_DIR / "07-model-response-curves.csv", index=False)
    perturbations.round(5).to_csv(TABLE_DIR / "07-case-perturbations.csv", index=False)
    selected.round(5).to_csv(TABLE_DIR / "07-selected-cases.csv", index=False, date_format="%Y-%m-%d")
    global_figure(coefficients, importance, responses)
    case_figure(selected, perturbations)

    print(f"Saved {DATA_PATH.relative_to(ROOT)}")
    for name in ["07-logistic-coefficients.csv", "07-permutation-importance.csv", "07-model-response-curves.csv", "07-case-perturbations.csv", "07-selected-cases.csv"]:
        print(f"Saved {(TABLE_DIR / name).relative_to(ROOT)}")
    print(f"Saved {GLOBAL_FIGURE.relative_to(ROOT)}")
    print(f"Saved {CASE_FIGURE.relative_to(ROOT)}")
    print(f"Held-out ROC AUC: {test_auc:.3f}")


if __name__ == "__main__":
    main()
