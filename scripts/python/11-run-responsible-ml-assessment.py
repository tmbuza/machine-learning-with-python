"""Generate subgroup, risk-register, and model-card evidence for MLPY-011."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "11-responsible-ml-data.csv"
TABLE_DIR = ROOT / "results" / "tables"
REPORT_PATH = ROOT / "results" / "reports" / "11-model-card.md"
FIGURE_PATH = ROOT / "results" / "figures" / "11-responsible-ml-assessment.png"
NUMERIC = ["tenure_months", "monthly_charge", "support_contacts", "usage_hours"]
CATEGORICAL = ["plan", "region"]
FEATURES = NUMERIC + CATEGORICAL


def create_data():
    rng = np.random.default_rng(SEED)
    count = 1500
    dates = rng.choice(pd.date_range("2026-01-31", periods=6, freq="ME"), count)
    groups = rng.choice(["Group A", "Group B", "Group C"], count, p=[0.52, 0.31, 0.17])
    plans = rng.choice(["Basic", "Standard", "Premium"], count, p=[0.43, 0.37, 0.20])
    regions = rng.choice(["North", "South", "East", "West"], count)
    tenure = rng.integers(2, 72, count)
    contacts = rng.poisson(1.2 + 0.35 * (groups == "Group C"))
    usage = rng.normal(43, 10, count) + np.where(plans == "Premium", 8, np.where(plans == "Basic", -6, 1))
    charge = np.select([plans == "Basic", plans == "Standard"], [35, 65], default=95) + rng.normal(0, 5, count)
    latent = rng.normal(0, 0.7, count)
    logit = -1.45 + latent + 0.55 * contacts - 0.02 * tenure - 0.04 * usage + 0.35 * (plans == "Basic") + 0.28 * (groups == "Group C")
    probability = 1 / (1 + np.exp(-logit))
    target = (rng.random(count) < probability).astype(int)
    data = pd.DataFrame({"customer_id": [f"C{n:05d}" for n in range(1, count + 1)], "snapshot_date": dates, "access_group": groups, "tenure_months": tenure, "monthly_charge": charge.round(2), "support_contacts": contacts, "usage_hours": usage.round(2), "plan": plans, "region": regions, "churned_next_month": target})
    data.loc[rng.choice(data.index, 32, replace=False), "monthly_charge"] = np.nan
    return data.sort_values("snapshot_date").reset_index(drop=True)


def build_pipeline():
    prep = ColumnTransformer([("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC), ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), CATEGORICAL)])
    return make_pipeline(prep, LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))


def metrics(frame):
    target, prediction = frame["observed"], frame["predicted"]
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    return {"rows": len(frame), "positive_cases": int(target.sum()), "base_rate": target.mean(), "selected_rows": int(prediction.sum()), "selection_rate": prediction.mean(), "true_positive_rate": tp / (tp + fn) if tp + fn else np.nan, "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan, "precision": precision_score(target, prediction, zero_division=0), "roc_auc": roc_auc_score(target, frame["probability"]) if target.nunique() == 2 else np.nan}


def main():
    for path in [DATA_PATH.parent, TABLE_DIR, REPORT_PATH.parent, FIGURE_PATH.parent]: path.mkdir(parents=True, exist_ok=True)
    data = create_data()
    data.to_csv(DATA_PATH, index=False, date_format="%Y-%m-%d")
    cutoff = data["snapshot_date"].max()
    train, test = data[data["snapshot_date"] < cutoff], data[data["snapshot_date"] == cutoff].copy()
    model = build_pipeline()
    model.fit(train[FEATURES], train["churned_next_month"])
    test["probability"] = model.predict_proba(test[FEATURES])[:, 1]
    test["predicted"] = (test["probability"] >= 0.55).astype(int)
    test["observed"] = test["churned_next_month"]
    overall = pd.DataFrame([metrics(test)])
    group_rows = []
    for group, frame in test.groupby("access_group"):
        group_rows.append({"access_group": group, **metrics(frame)})
    groups = pd.DataFrame(group_rows)
    reference = groups.sort_values("rows", ascending=False).iloc[0]
    comparisons = groups.copy()
    for metric in ["selection_rate", "true_positive_rate", "false_positive_rate", "precision"]:
        comparisons[f"{metric}_difference_from_{reference['access_group'].replace(' ', '_')}"] = comparisons[metric] - reference[metric]
        comparisons[f"{metric}_ratio_to_{reference['access_group'].replace(' ', '_')}"] = comparisons[metric] / reference[metric] if reference[metric] else np.nan

    risks = pd.DataFrame([
        ["R01", "Historical labels may encode unequal service processes", "Problem and data", "Medium", "High", "Review label provenance and subgroup outcomes", "Data owner", "Open"],
        ["R02", "Recall and false-positive rates differ across audit groups", "Evaluation", "Medium", "High", "Review thresholds, data quality, harms, and uncertainty", "Model owner", "Open"],
        ["R03", "Predictions or explanations could expose personal information", "Use", "Medium", "High", "Minimize outputs; restrict access and retention", "Privacy owner", "Controlled"],
        ["R04", "Score may be used outside voluntary retention support", "Use", "Medium", "High", "Document prohibited uses and require use-case approval", "Business owner", "Controlled"],
        ["R05", "Performance may become stale after population change", "Monitoring", "Medium", "Medium", "Monitor drift and mature-label performance", "Model owner", "Controlled"],
        ["R06", "Reviewers may follow scores without meaningful judgment", "Human oversight", "Medium", "High", "Train reviewers; audit overrides and appeals", "Operations owner", "Open"],
    ], columns=["risk_id", "risk_statement", "lifecycle_stage", "likelihood", "impact", "control", "owner", "status"])
    controls = pd.DataFrame([
        ["Purpose limitation", "Use only for voluntary retention-support prioritization", "Pre-release and annual review"],
        ["Audit-field separation", "Exclude access_group from model inputs and ordinary prediction outputs", "Every release"],
        ["Human review", "No adverse automated action; reviewer can override", "Every decision"],
        ["Appeal", "Record and investigate disputed actions", "Ongoing"],
        ["Privacy", "Role-based access, minimal logs, retention schedule", "Quarterly review"],
        ["Stop condition", "Pause on schema failure, material subgroup harm, or stale performance", "Continuous"],
    ], columns=["control_area", "required_control", "review_frequency"])
    overall.round(4).to_csv(TABLE_DIR / "11-overall-performance.csv", index=False)
    groups.round(4).to_csv(TABLE_DIR / "11-group-performance.csv", index=False)
    comparisons.round(4).to_csv(TABLE_DIR / "11-fairness-comparisons.csv", index=False)
    risks.to_csv(TABLE_DIR / "11-model-risk-register.csv", index=False)
    controls.to_csv(TABLE_DIR / "11-governance-controls.csv", index=False)

    model_card = f"""# Model Card: Customer Retention Support Model\n\n## Purpose\nPrioritize customers for voluntary retention-support review. Model version: 1.0.0.\n\n## Prohibited uses\nDo not deny service, change prices, or take adverse action automatically.\n\n## Evaluation\nChronological final-month holdout with {len(test)} rows. ROC AUC: {overall.iloc[0]['roc_auc']:.3f}. Decision threshold: 0.55.\n\n## Feature policy\nUses {', '.join(FEATURES)}. `access_group` is retained for approved auditing only and excluded from model features.\n\n## Human oversight\nEvery action requires review. Reviewers may override, must record reasons, and must provide an appeal route.\n\n## Limitations\nSynthetic educational data; group estimates vary with sample size; predictive associations are not causal; results are not approval for real-world use.\n\n## Monitoring\nMonitor input quality, drift, mature-label performance, group metrics, overrides, complaints, and incidents. Pause use when stop conditions are met.\n"""
    REPORT_PATH.write_text(model_card, encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    x = np.arange(len(groups)); labels = groups["access_group"]
    axes[0, 0].bar(labels, groups["rows"], color="#4C78A8"); axes[0, 0].set_title("Evaluation rows by audit group"); axes[0, 0].set_ylabel("Rows")
    width = 0.36
    axes[0, 1].bar(x-width/2, groups["base_rate"], width, label="Observed outcome rate", color="#4C78A8")
    axes[0, 1].bar(x+width/2, groups["selection_rate"], width, label="Selection rate", color="#F28E2B")
    axes[0, 1].set_xticks(x, labels); axes[0, 1].set_title("Outcomes and model selections"); axes[0, 1].legend(frameon=False)
    axes[1, 0].bar(x-width/2, groups["true_positive_rate"], width, label="True-positive rate", color="#59A14F")
    axes[1, 0].bar(x+width/2, groups["false_positive_rate"], width, label="False-positive rate", color="#E15759")
    axes[1, 0].set_xticks(x, labels); axes[1, 0].set_ylim(0, 1); axes[1, 0].set_title("Error-rate trade-offs"); axes[1, 0].legend(frameon=False)
    axes[1, 1].bar(labels, groups["precision"], color="#B279A2"); axes[1, 1].set_ylim(0, 1); axes[1, 1].set_title("Precision with denominator context")
    for position, row in enumerate(groups.itertuples()): axes[1, 1].text(position, row.precision + .03, f"n={row.rows}\npos={row.positive_cases}", ha="center", fontsize=9)
    fig.suptitle("Responsible review combines metrics, context, and governance", fontsize=16, fontweight="bold")
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight"); plt.close(fig)
    for name in ["11-overall-performance.csv", "11-group-performance.csv", "11-fairness-comparisons.csv", "11-model-risk-register.csv", "11-governance-controls.csv"]: print(f"Saved {(TABLE_DIR / name).relative_to(ROOT)}")
    print(f"Saved {REPORT_PATH.relative_to(ROOT)}"); print(f"Saved {FIGURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__": main()
