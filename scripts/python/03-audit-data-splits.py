"""Generate and audit candidate splitting strategies for MLPY-003."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "03-customer-snapshots.csv"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_PATH = ROOT / "results" / "figures" / "03-splitting-strategies.png"


def create_snapshots() -> pd.DataFrame:
    """Create repeated customer snapshots with a reproducible churn target."""
    rng = np.random.default_rng(SEED)
    customer_count = 180
    dates = pd.date_range("2026-01-31", periods=4, freq="ME")
    customer_ids = [f"C{number:04d}" for number in range(1, customer_count + 1)]
    base_risk = rng.normal(0, 0.8, customer_count)
    plan = rng.choice(["Basic", "Standard", "Premium"], customer_count, p=[0.4, 0.4, 0.2])
    region = rng.choice(["North", "South", "East", "West"], customer_count)

    rows = []
    row_id = 1
    for month_number, snapshot_date in enumerate(dates):
        for position, customer_id in enumerate(customer_ids):
            tenure = rng.integers(1, 60) + month_number
            support_contacts = rng.poisson(1.2 + 0.25 * max(base_risk[position], 0))
            monthly_charge = {
                "Basic": 35,
                "Standard": 65,
                "Premium": 95,
            }[plan[position]] + rng.normal(0, 6)
            logit = (
                -2.1
                + base_risk[position]
                + 0.28 * support_contacts
                + 0.18 * month_number
                - 0.018 * tenure
                + (0.22 if plan[position] == "Basic" else 0)
            )
            probability = 1 / (1 + np.exp(-logit))
            churned = int(rng.random() < probability)
            rows.append(
                {
                    "row_id": row_id,
                    "customer_id": customer_id,
                    "snapshot_date": snapshot_date,
                    "plan": plan[position],
                    "region": region[position],
                    "tenure_months": int(tenure),
                    "monthly_charge": round(float(monthly_charge), 2),
                    "support_contacts": int(support_contacts),
                    "churned_next_month": churned,
                }
            )
            row_id += 1

    return pd.DataFrame(rows)


def assign_splits(data: pd.DataFrame) -> dict[str, tuple[pd.Index, pd.Index]]:
    """Return index-label assignments for three candidate designs."""
    random_train, random_test = train_test_split(
        data.index,
        test_size=0.25,
        stratify=data["churned_next_month"],
        random_state=SEED,
    )

    grouped = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    group_train_position, group_test_position = next(
        grouped.split(data, groups=data["customer_id"])
    )
    group_train = data.index[group_train_position]
    group_test = data.index[group_test_position]

    cutoff = data["snapshot_date"].max()
    time_train = data.index[data["snapshot_date"] < cutoff]
    time_test = data.index[data["snapshot_date"] >= cutoff]

    return {
        "Stratified random": (pd.Index(random_train), pd.Index(random_test)),
        "Customer grouped": (pd.Index(group_train), pd.Index(group_test)),
        "Chronological": (pd.Index(time_train), pd.Index(time_test)),
    }


def audit_split(
    name: str, data: pd.DataFrame, train_index: pd.Index, test_index: pd.Index
) -> dict[str, object]:
    """Calculate diagnostics for one split."""
    train = data.loc[train_index]
    test = data.loc[test_index]
    overlap = set(train["customer_id"]) & set(test["customer_id"])
    train_max = train["snapshot_date"].max()
    test_min = test["snapshot_date"].min()
    return {
        "strategy": name,
        "training_rows": len(train),
        "test_rows": len(test),
        "training_positive_rate": round(train["churned_next_month"].mean(), 4),
        "test_positive_rate": round(test["churned_next_month"].mean(), 4),
        "customer_overlap_count": len(overlap),
        "training_max_date": train_max.date().isoformat(),
        "test_min_date": test_min.date().isoformat(),
        "strict_time_order": bool(train_max < test_min),
    }


def build_assignment_table(
    data: pd.DataFrame, splits: dict[str, tuple[pd.Index, pd.Index]]
) -> pd.DataFrame:
    """Record every row's partition under each candidate design."""
    assignments = data[["row_id", "customer_id", "snapshot_date", "churned_next_month"]].copy()
    column_names = {
        "Stratified random": "stratified_random_partition",
        "Customer grouped": "customer_grouped_partition",
        "Chronological": "chronological_partition",
    }
    for name, (train_index, test_index) in splits.items():
        assignments[column_names[name]] = "unassigned"
        assignments.loc[train_index, column_names[name]] = "train"
        assignments.loc[test_index, column_names[name]] = "test"
        assert not (assignments[column_names[name]] == "unassigned").any()
    return assignments


def plot_audit(
    data: pd.DataFrame,
    splits: dict[str, tuple[pd.Index, pd.Index]],
    audit: pd.DataFrame,
) -> None:
    """Create a four-panel comparison of splitting strategies."""
    labels = ["Random", "Grouped", "Chronological"]
    colors = ["#2C7FB8", "#41AB5D", "#F28E2B"]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)

    width = 0.34
    axes[0, 0].bar(x - width / 2, audit["training_positive_rate"] * 100, width, label="Train", color="#4C78A8")
    axes[0, 0].bar(x + width / 2, audit["test_positive_rate"] * 100, width, label="Test", color="#E45756")
    axes[0, 0].set_title("Target rate by partition")
    axes[0, 0].set_ylabel("Positive class (%)")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].bar(labels, audit["customer_overlap_count"], color=colors)
    axes[0, 1].set_title("Customers appearing in both partitions")
    axes[0, 1].set_ylabel("Overlapping customers")
    for position, value in enumerate(audit["customer_overlap_count"]):
        axes[0, 1].text(position, value + 3, str(value), ha="center", fontweight="bold")

    dates = sorted(data["snapshot_date"].unique())
    for strategy_position, (name, (train_index, test_index)) in enumerate(splits.items()):
        for date in dates:
            date_rows = data.index[data["snapshot_date"] == date]
            test_share = len(date_rows.intersection(test_index)) / len(date_rows)
            axes[1, 0].scatter(
                pd.Timestamp(date), strategy_position,
                s=90 + test_share * 260,
                color="#E45756" if test_share > 0.5 else "#4C78A8",
                alpha=0.85,
                edgecolor="white",
            )
    axes[1, 0].set_title("Time coverage (marker size reflects test share)")
    axes[1, 0].set_yticks(range(3), labels)
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].grid(axis="x", alpha=0.25)

    axes[1, 1].bar(x - width / 2, audit["training_rows"], width, label="Train", color="#4C78A8")
    axes[1, 1].bar(x + width / 2, audit["test_rows"], width, label="Test", color="#E45756")
    axes[1, 1].set_title("Partition sizes")
    axes[1, 1].set_ylabel("Rows")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].legend(frameon=False)

    fig.suptitle("Split design changes what 'unseen data' means", fontsize=16, fontweight="bold")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate data, split audits, assignments, and the comparison figure."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    data = create_snapshots()
    data.to_csv(DATA_PATH, index=False, date_format="%Y-%m-%d")
    splits = assign_splits(data)

    audit = pd.DataFrame(
        audit_split(name, data, train_index, test_index)
        for name, (train_index, test_index) in splits.items()
    )

    grouped_train, grouped_test = splits["Customer grouped"]
    assert set(data.loc[grouped_train, "customer_id"]).isdisjoint(
        set(data.loc[grouped_test, "customer_id"])
    )
    time_train, time_test = splits["Chronological"]
    assert data.loc[time_train, "snapshot_date"].max() < data.loc[time_test, "snapshot_date"].min()

    audit.to_csv(TABLE_DIR / "03-split-audit.csv", index=False)
    build_assignment_table(data, splits).to_csv(
        TABLE_DIR / "03-split-assignments.csv", index=False, date_format="%Y-%m-%d"
    )
    plot_audit(data, splits, audit)

    print(f"Saved {DATA_PATH.relative_to(ROOT)}")
    print(f"Saved {(TABLE_DIR / '03-split-audit.csv').relative_to(ROOT)}")
    print(f"Saved {(TABLE_DIR / '03-split-assignments.csv').relative_to(ROOT)}")
    print(f"Saved {FIGURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
