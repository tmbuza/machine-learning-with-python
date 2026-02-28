# Machine Learning with Python

**CDI Free Track**

Workflow-driven introduction to machine learning with Python.\
Part of the Complex Data Insights (CDI) learning ecosystem.

------------------------------------------------------------------------

## About This Guide

Machine learning is often presented as a collection of algorithms.

This guide teaches it as a structured workflow:

Design → Data → Model → Evaluation → Calibration → Interpretation

You will learn how to:

-   Frame regression and classification problems clearly\
-   Prepare data without leakage\
-   Train baseline predictive models\
-   Evaluate models with appropriate metrics\
-   Interpret results responsibly

This is the **free foundational track**.\
Advanced modeling, tuning, and deployment will appear in the premium
track.

------------------------------------------------------------------------

## Structure

This guide is built using **Quarto (book format)**.

-   Navigation is handled by `_quarto.yml`
-   `index.qmd` is cover-only
-   Lessons follow the CDI universal ID convention:

```{=html}
<!-- -->
```
    MLPY-F-L01
    MLPY-F-L02
    MLPY-F-L03

Format:

    DOMAIN-TRACK-TYPE##

Example:

    MLPY-F-L01

------------------------------------------------------------------------

## Repository Structure

    machine-learning-with-python/
    ├── _quarto.yml
    ├── index.qmd
    ├── 01-preface-and-setup.qmd
    ├── 02-ml-thinking-and-problem-types.qmd
    ...
    ├── cdi_ml/              # Helper utilities
    ├── data/                # Raw, processed, and ML-ready datasets
    ├── scripts/             # Environment + dataset generator
    ├── figures/             # Auto-saved figures
    └── docs/                # Rendered output (GitHub Pages)

------------------------------------------------------------------------

## Quick Start

### 1. Create environment

``` bash
bash scripts/setup-env.sh
source .venv/bin/activate
```

### 2. Generate dataset

``` bash
python scripts/make-cdi-customer-churn.py
```

### 3. Render book

``` bash
bash scripts/build-all.sh
```

The rendered site will appear in:

    docs/

------------------------------------------------------------------------

## Versioning

This repository follows semantic versioning.

-   v0.0.1 → Scaffold initialized\
-   v0.x.x → Draft development\
-   v1.0.0 → First stable release

------------------------------------------------------------------------

## License

To be defined.

------------------------------------------------------------------------

## Complex Data Insights (CDI)

CDI provides structured, domain-focused learning guides across:

-   Data Science\
-   Visualization\
-   Machine Learning\
-   Applied Bioinformatics

This repository is part of the public free track.
