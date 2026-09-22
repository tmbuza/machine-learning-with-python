#!/usr/bin/env python3
"""Generate the ML problem-types figure used in Chapter 01."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEED = 42
OUTPUT = Path("results/figures/01-ml-problem-types.png")


def main() -> None:
    """Create a four-panel visual summary of common ML problem types."""
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle("Machine-learning problem types", fontsize=18, fontweight="bold")

    # Regression: predict a continuous value.
    x_reg = np.linspace(1, 10, 34)
    y_reg = 1.8 * x_reg + 4 + rng.normal(0, 2.2, x_reg.size)
    coef = np.polyfit(x_reg, y_reg, 1)
    axes[0, 0].scatter(x_reg, y_reg, color="#2878B5", alpha=0.78, edgecolor="white")
    axes[0, 0].plot(x_reg, np.polyval(coef, x_reg), color="#D1495B", linewidth=2.5)
    axes[0, 0].set(title="Regression", xlabel="Feature", ylabel="Continuous outcome")

    # Classification: predict a category or class probability.
    class_a = rng.normal(loc=(2.8, 3.0), scale=(0.75, 0.85), size=(35, 2))
    class_b = rng.normal(loc=(6.2, 6.0), scale=(0.85, 0.8), size=(35, 2))
    axes[0, 1].scatter(class_a[:, 0], class_a[:, 1], label="Class A", color="#59A14F", alpha=0.78)
    axes[0, 1].scatter(class_b[:, 0], class_b[:, 1], label="Class B", color="#F28E2B", alpha=0.78)
    axes[0, 1].plot([1.5, 7.5], [7.5, 1.5], linestyle="--", color="#555555", linewidth=2)
    axes[0, 1].set(title="Classification", xlabel="Feature 1", ylabel="Feature 2")
    axes[0, 1].legend(frameon=False)

    # Ranking: order cases by a model score.
    cases = np.array(["Case B", "Case E", "Case A", "Case D", "Case C"])
    scores = np.array([0.93, 0.81, 0.68, 0.51, 0.29])
    positions = np.arange(len(cases))
    axes[1, 0].barh(positions, scores, color="#4E79A7")
    axes[1, 0].set_yticks(positions, cases)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set(title="Ranking", xlabel="Priority score", xlim=(0, 1))
    for pos, score in zip(positions, scores):
        axes[1, 0].text(score + 0.02, pos, f"{score:.2f}", va="center", fontsize=9)

    # Clustering: find structure without target labels.
    centers = np.array([[2.0, 5.5], [5.0, 2.0], [7.0, 6.0]])
    colors = ["#B07AA1", "#76B7B2", "#E15759"]
    for center, color in zip(centers, colors):
        points = rng.normal(loc=center, scale=(0.65, 0.65), size=(28, 2))
        axes[1, 1].scatter(points[:, 0], points[:, 1], color=color, alpha=0.75)
    axes[1, 1].set(title="Clustering", xlabel="Feature 1", ylabel="Feature 2")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.18, linewidth=0.7)

    fig.text(
        0.5,
        0.015,
        "Problem framing determines the output a model must produce.",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
