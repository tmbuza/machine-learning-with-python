"""Run expanding-window validation and pipeline tuning for MLPY-005."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "05-customer-validation-table.csv"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_PATH = ROOT / "results" / "figures" / "05-cross-validation-and-tuning.png"

NUMERIC_FEATURES = ["tenure_months", "monthly_charge", "support_contacts", "usage_hours"]
CATEGORICAL_FEATURES = ["plan", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def create_data() -> pd.DataFrame:
    """Create eight monthly snapshots with moderate temporal drift."""
    rng = np.random.default_rng(SEED)
    customer_count = 280
    dates = pd.date_range("2026-01-31", periods=8, freq="ME")
    identifiers = np.array([f"C{number:04d}" for number in range(1, customer_count + 1)])
    plans = rng.choice(["Basic", "Standard", "Premium"], customer_count, p=[0.43, 0.37, 0.20])
    regions = rng.choice(["North", "South", "East", "West"], customer_count)
    latent_risk = rng.normal(0, 0.7, customer_count)
    latent_usage = rng.normal(43, 9, customer_count)

    rows = []
    for month, date in enumerate(dates):
        for position, customer_id in enumerate(identifiers):
            tenure = int(rng.integers(2, 70) + month)
            contacts = int(rng.poisson(1.1 + 0.45 * max(latent_risk[position], 0)))
            charge = {"Basic": 36, "Standard": 66, "Premium": 96}[plans[position]] + rng.normal(0, 5)
            usage = (
                latent_usage[position]
                + {"Basic": -7, "Standard": 1, "Premium": 9}[plans[position]]
                + 0.7 * month
                + rng.normal(0, 5)
            )
            logit = (
                -1.45
                + latent_risk[position]
                + 0.52 * contacts
                - 0.017 * tenure
                - 0.038 * usage
                + 0.08 * month
                + (0.38 if plans[position] == "Basic" else 0)
            )
            probability = 1 / (1 + np.exp(-logit))
            rows.append(
                {
                    "customer_id": customer_id,
                    "snapshot_date": date,
                    "plan": plans[position],
                    "region": regions[position],
                    "tenure_months": tenure,
                    "monthly_charge": round(float(charge), 2),
                    "support_contacts": contacts,
                    "usage_hours": round(max(float(usage), 0), 2),
                    "churned_next_month": int(rng.random() < probability),
                }
            )

    data = pd.DataFrame(rows)
    data.loc[rng.choice(data.index, 42, replace=False), "monthly_charge"] = np.nan
    data.loc[rng.choice(data.index, 24, replace=False), "plan"] = np.nan
    return data


def make_preprocessor() -> ColumnTransformer:
    """Return a fresh mixed-type preprocessor."""
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
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


def make_folds(development: pd.DataFrame):
    """Create three expanding folds using complete validation months."""
    months = sorted(development["snapshot_date"].unique())
    validation_months = months[-3:]
    folds = []
    records = []
    for fold_number, validation_month in enumerate(validation_months, start=1):
        train_index = np.flatnonzero(development["snapshot_date"].to_numpy() < validation_month)
        validation_index = np.flatnonzero(development["snapshot_date"].to_numpy() == validation_month)
        assert development.iloc[train_index]["snapshot_date"].max() < development.iloc[validation_index]["snapshot_date"].min()
        folds.append((train_index, validation_index))
        records.append(
            {
                "fold": fold_number,
                "training_start": development.iloc[train_index]["snapshot_date"].min().date().isoformat(),
                "training_end": development.iloc[train_index]["snapshot_date"].max().date().isoformat(),
                "validation_month": pd.Timestamp(validation_month).date().isoformat(),
                "training_rows": len(train_index),
                "validation_rows": len(validation_index),
                "training_positive_rate": development.iloc[train_index]["churned_next_month"].mean(),
                "validation_positive_rate": development.iloc[validation_index]["churned_next_month"].mean(),
            }
        )
    return folds, pd.DataFrame(records)


def candidate_pipelines():
    """Return baseline and simple candidate workflows."""
    return {
        "Prior baseline": make_pipeline(make_preprocessor(), DummyClassifier(strategy="prior")),
        "Logistic regression": make_pipeline(
            make_preprocessor(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
        ),
        "Untuned tree": make_pipeline(
            make_preprocessor(),
            DecisionTreeClassifier(
                class_weight="balanced", max_depth=4, min_samples_leaf=25, random_state=SEED
            ),
        ),
    }


def cross_validate_candidates(X, y, folds) -> pd.DataFrame:
    """Save fold-level scores for all reference workflows."""
    rows = []
    for model_name, workflow in candidate_pipelines().items():
        scores = cross_validate(
            workflow,
            X,
            y,
            cv=folds,
            scoring={"roc_auc": "roc_auc", "balanced_accuracy": "balanced_accuracy"},
        )
        for fold_number, (auc, balanced) in enumerate(
            zip(scores["test_roc_auc"], scores["test_balanced_accuracy"]), start=1
        ):
            rows.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "roc_auc": auc,
                    "balanced_accuracy": balanced,
                }
            )
    return pd.DataFrame(rows)


def tune_tree(X, y, folds):
    """Tune a tree pipeline inside expanding temporal folds."""
    workflow = make_pipeline(
        make_preprocessor(),
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
    )
    grid = {
        "decisiontreeclassifier__max_depth": [2, 3, 4, 6],
        "decisiontreeclassifier__min_samples_leaf": [10, 25, 50],
    }
    search = GridSearchCV(
        workflow,
        param_grid=grid,
        scoring="roc_auc",
        cv=folds,
        refit=True,
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(X, y)
    results = pd.DataFrame(search.cv_results_)[
        [
            "param_decisiontreeclassifier__max_depth",
            "param_decisiontreeclassifier__min_samples_leaf",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
    ].rename(
        columns={
            "param_decisiontreeclassifier__max_depth": "max_depth",
            "param_decisiontreeclassifier__min_samples_leaf": "min_samples_leaf",
            "mean_test_score": "mean_validation_roc_auc",
            "std_test_score": "sd_validation_roc_auc",
            "rank_test_score": "rank",
        }
    )
    return search, results.sort_values("rank")


def final_evaluation(models, X_development, y_development, X_test, y_test):
    """Fit selected workflows and evaluate once on final test data."""
    rows = []
    probabilities = {}
    for name, workflow in models.items():
        workflow.fit(X_development, y_development)
        prediction = workflow.predict(X_test)
        probability = workflow.predict_proba(X_test)[:, 1]
        probabilities[name] = probability
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, prediction),
                "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
                "roc_auc": roc_auc_score(y_test, probability),
            }
        )
    return pd.DataFrame(rows), probabilities


def make_figure(data, fold_table, cv_results, tuning_results, final_results, final_probabilities, y_test):
    """Create a four-panel validation and tuning figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {"Prior baseline": "#9C9C9C", "Logistic regression": "#4C78A8", "Untuned tree": "#F28E2B"}

    months = [pd.Timestamp(value).strftime("%b") for value in sorted(data["snapshot_date"].unique())]
    fold_matrix = np.zeros((3, len(months)))
    for row in fold_table.itertuples():
        train_end = pd.Timestamp(row.training_end).strftime("%b")
        valid = pd.Timestamp(row.validation_month).strftime("%b")
        fold_matrix[row.fold - 1, : months.index(train_end) + 1] = 1
        fold_matrix[row.fold - 1, months.index(valid)] = 2
    axes[0, 0].imshow(fold_matrix, aspect="auto", cmap=plt.matplotlib.colors.ListedColormap(["white", "#4C78A8", "#F28E2B"]), vmin=0, vmax=2)
    axes[0, 0].set_xticks(range(len(months)), months)
    axes[0, 0].set_yticks(range(3), ["Fold 1", "Fold 2", "Fold 3"])
    axes[0, 0].set_title("Expanding validation design")
    axes[0, 0].set_xlabel("Blue = train, orange = validate; Aug = final test")

    for model, group in cv_results.groupby("model", sort=False):
        axes[0, 1].plot(group["fold"], group["roc_auc"], marker="o", linewidth=2, label=model, color=colors[model])
    axes[0, 1].set_title("Validation ROC AUC by fold")
    axes[0, 1].set_xlabel("Fold")
    axes[0, 1].set_ylabel("ROC AUC")
    axes[0, 1].set_xticks([1, 2, 3])
    axes[0, 1].set_ylim(0.4, 0.85)
    axes[0, 1].legend(frameon=False)

    surface = tuning_results.pivot(index="max_depth", columns="min_samples_leaf", values="mean_validation_roc_auc").sort_index()
    image = axes[1, 0].imshow(surface.values, cmap="YlGnBu", aspect="auto", vmin=surface.values.min() - 0.01, vmax=surface.values.max() + 0.01)
    axes[1, 0].set_xticks(range(len(surface.columns)), surface.columns)
    axes[1, 0].set_yticks(range(len(surface.index)), surface.index)
    axes[1, 0].set_xlabel("Minimum samples per leaf")
    axes[1, 0].set_ylabel("Maximum depth")
    axes[1, 0].set_title("Tree tuning: mean validation ROC AUC")
    for row in range(surface.shape[0]):
        for column in range(surface.shape[1]):
            axes[1, 0].text(column, row, f"{surface.iloc[row, column]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    final_names = ["Prior baseline", "Logistic regression", "Tuned tree"]
    final_colors = ["#9C9C9C", "#4C78A8", "#41AB5D"]
    for name, color in zip(final_names, final_colors):
        false_positive, true_positive, _ = roc_curve(y_test, final_probabilities[name])
        auc = final_results.set_index("model").loc[name, "roc_auc"]
        axes[1, 1].plot(false_positive, true_positive, linewidth=2, color=color, label=f"{name} ({auc:.3f})")
    axes[1, 1].plot([0, 1], [0, 1], linestyle="--", color="#333333", linewidth=1)
    axes[1, 1].set_title("Final test ROC curves")
    axes[1, 1].set_xlabel("False-positive rate")
    axes[1, 1].set_ylabel("True-positive rate")
    axes[1, 1].legend(frameon=False, loc="lower right")

    fig.suptitle("Tune inside development data; evaluate the final test once", fontsize=16, fontweight="bold")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate data, validate candidates, tune a tree, and test once."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    data = create_data()
    data.to_csv(DATA_PATH, index=False, date_format="%Y-%m-%d")

    final_month = data["snapshot_date"].max()
    development = data[data["snapshot_date"] < final_month].reset_index(drop=True)
    test = data[data["snapshot_date"] == final_month].reset_index(drop=True)
    assert development["snapshot_date"].max() < test["snapshot_date"].min()
    folds, fold_table = make_folds(development)

    X_development = development[FEATURES]
    y_development = development["churned_next_month"]
    X_test = test[FEATURES]
    y_test = test["churned_next_month"]

    cv_results = cross_validate_candidates(X_development, y_development, folds)
    search, tuning_results = tune_tree(X_development, y_development, folds)
    final_models = {
        "Prior baseline": candidate_pipelines()["Prior baseline"],
        "Logistic regression": candidate_pipelines()["Logistic regression"],
        "Tuned tree": search.best_estimator_,
    }
    final_results, final_probabilities = final_evaluation(
        final_models, X_development, y_development, X_test, y_test
    )

    fold_table.round(4).to_csv(TABLE_DIR / "05-validation-folds.csv", index=False)
    cv_results.round(4).to_csv(TABLE_DIR / "05-cross-validation-results.csv", index=False)
    tuning_results.round(4).to_csv(TABLE_DIR / "05-tuning-results.csv", index=False)
    final_results.round(4).to_csv(TABLE_DIR / "05-final-test-results.csv", index=False)
    make_figure(data, fold_table, cv_results, tuning_results, final_results, final_probabilities, y_test)

    print(f"Saved {DATA_PATH.relative_to(ROOT)}")
    for name in ["05-validation-folds.csv", "05-cross-validation-results.csv", "05-tuning-results.csv", "05-final-test-results.csv"]:
        print(f"Saved {(TABLE_DIR / name).relative_to(ROOT)}")
    print(f"Saved {FIGURE_PATH.relative_to(ROOT)}")
    print(f"Best tree parameters: {search.best_params_}")


if __name__ == "__main__":
    main()
