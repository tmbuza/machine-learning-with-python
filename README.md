# Machine Learning with Python

**A structured workflow for predictive modeling**

Machine Learning with Python is a free, professional learning guide from [Complex Data Insights](https://complexdatainsights.com/). It teaches machine learning as a complete, reproducible workflow rather than as a disconnected collection of algorithms.

The guide moves from problem framing and data preparation through model development, evaluation, interpretation, persistence, batch prediction, monitoring, responsible use, and an end-to-end case study.

**Published guide:** <https://machine-learning.complexdatainsights.com/>

## What this guide emphasizes

The central workflow is:

> Define the decision → establish the data contract → split honestly → build a baseline → validate carefully → diagnose failures → interpret cautiously → release reproducibly → monitor continuously → govern responsibly.

Learners work through how to:

- translate a real-world need into a well-defined machine-learning problem;
- distinguish regression, classification, ranking, clustering, anomaly detection, and forecasting tasks;
- prepare tabular data without introducing leakage;
- design training, validation, and test splits that reflect the intended use;
- establish meaningful baselines before adding complexity;
- combine preprocessing and estimation safely with scikit-learn pipelines;
- use cross-validation and hyperparameter tuning appropriately;
- select metrics and decision thresholds that match practical costs;
- diagnose model errors, calibration, and subgroup performance;
- interpret global and local model behavior without making unsupported causal claims;
- persist complete prediction workflows reproducibly;
- validate and audit batch predictions;
- monitor data quality, drift, model performance, and operational outcomes;
- document responsible-use constraints and governance decisions; and
- complete an end-to-end machine-learning project.

## Guide structure

| File | ID | Topic |
|---|---|---|
| `00-preface.qmd` | MLPY-000 | Guide orientation, environment setup, and reproducible workflow |
| `01-ml-thinking-and-problem-types.qmd` | MLPY-001 | ML thinking and problem types |
| `02-data-preparation-for-ml.qmd` | MLPY-002 | Data preparation for machine learning |
| `03-data-splitting-and-leakage.qmd` | MLPY-003 | Data splitting and leakage prevention |
| `04-baselines-and-first-models.qmd` | MLPY-004 | Baselines and first models |
| `05-cross-validation-and-hyperparameter-tuning.qmd` | MLPY-005 | Cross-validation and hyperparameter tuning |
| `06-model-evaluation-and-diagnostics.qmd` | MLPY-006 | Model evaluation and diagnostics |
| `07-model-interpretation-and-explainability.qmd` | MLPY-007 | Model interpretation and explainability |
| `08-model-persistence-and-reproducible-prediction.qmd` | MLPY-008 | Model persistence and reproducible prediction |
| `09-batch-prediction-and-operational-validation.qmd` | MLPY-009 | Batch prediction and operational validation |
| `10-monitoring-drift-and-model-performance.qmd` | MLPY-010 | Monitoring, drift, and model performance |
| `11-responsible-machine-learning-and-governance.qmd` | MLPY-011 | Responsible machine learning and governance |
| `12-end-to-end-machine-learning-case-study.qmd` | MLPY-012 | End-to-end machine-learning case study |
| `99-appendix.qmd` | MLPY-APP | Professional reference appendix |
| `99-references.qmd` | MLPY-REF | Generated bibliography |

The cover page and preface are unnumbered. The first learning chapter therefore begins as Chapter 1, and its figures begin with Figure 1.1.

## Repository structure

```text
machine-learning-with-python/
├── _quarto.yml
├── index.qmd
├── 00-preface.qmd
├── 01-ml-thinking-and-problem-types.qmd
├── ...
├── 12-end-to-end-machine-learning-case-study.qmd
├── 99-appendix.qmd
├── 99-references.qmd
├── requirements.txt
├── assets/
│   ├── css/
│   └── images/
├── library/
│   └── references.bib
├── scripts/
│   ├── bash/
│   └── python/
├── data/
├── artifacts/
├── results/
├── docs/
└── .githooks/
    ├── install.sh
    ├── pre-commit
    └── README.md
```

## Code and execution policy

The guide deliberately separates instructional prose from maintained executable programs:

- QMD files contain non-executable examples using ordinary `python` and `bash` fences.
- Complete Python programs live in `scripts/python/`.
- Bash launchers live in `scripts/bash/`.
- Numbered scripts correspond to numbered chapters.
- Generated figures and analytical outputs are written under `results/`.
- Persisted model-related outputs are written under `artifacts/` where appropriate.
- The rendered Quarto site is written to `docs/` for GitHub Pages.

This separation keeps rendering predictable while giving learners reproducible programs they can run, inspect, and adapt.

## Requirements

The standard CDI development environment uses:

- Python 3.12;
- Quarto;
- Visual Studio Code;
- the VS Code Python extension; and
- a repository-specific `.venv` environment.

Python dependencies are declared in `requirements.txt`.

## Quick start

### 1. Clone and enter the repository

```bash
git clone <repository-url>
cd machine-learning-with-python
```

### 2. Create the project environment

```bash
bash scripts/bash/00-create-venv.sh
```

The script creates `.venv`, upgrades the packaging tools, and installs the dependencies from `requirements.txt`.

### 3. Select the interpreter in VS Code

Open the repository in VS Code, run **Python: Select Interpreter**, and select:

```text
machine-learning-with-python/.venv/bin/python
```

Open a new integrated terminal after selecting the interpreter.

### 4. Verify the environment

```bash
python -c "import sys; print(sys.executable)"
```

The path should point to this repository's `.venv`.

Verify the core packages:

```bash
python -c "import sklearn, pandas, numpy, matplotlib; print('Environment ready:', sklearn.__version__)"
```

### 5. Run a chapter workflow

For Chapter 1:

```bash
bash scripts/bash/01-generate-ml-problem-types.sh
```

Run other chapter workflows through their corresponding numbered Bash launchers. The launchers resolve the repository root and call the environment's Python interpreter directly.

To list the available workflows:

```bash
find scripts/bash -maxdepth 1 -type f -name '*.sh' | sort
```

### 6. Render the guide

```bash
quarto render
```

Open the rendered site on macOS:

```bash
open docs/index.html
```

## Rebuilding generated outputs

The contents of `results/`, `artifacts/`, and `docs/` are reproducible outputs. If generated results are unavailable, run the numbered chapter workflows in ascending order before rendering the guide.

```bash
set -euo pipefail

while IFS= read -r script; do
  echo "Running ${script}"
  bash "${script}"
done < <(
  find scripts/bash \
    -maxdepth 1 \
    -type f \
    -name '[0-9][0-9]-*.sh' \
    ! -name '00-create-venv.sh' \
    | sort
)
```

Then render the book:

```bash
quarto render
```

## Citations

Bibliographic records are stored in:

```text
library/references.bib
```

The project configuration points Quarto to that database:

```yaml
bibliography: library/references.bib
```

Inline citations use Pandoc citation keys such as:

```markdown
Cross-validation estimates performance on unseen data [@Kohavi1995; @Kuhn2013].
```

Quarto collects cited records in `99-references.qmd`.

## Git hooks and guide dates

The repository includes a version-controlled pre-commit hook that updates the guide date in `_quarto.yml` when meaningful source files change.

Configure the hooks once after cloning:

```bash
bash .githooks/install.sh
```

Confirm the local configuration:

```bash
git config --local --get core.hooksPath
```

Expected output:

```text
.githooks
```

Git does not transfer local repository configuration during cloning, so the installer must be run once for every new clone. See `.githooks/README.md` for details.

## Contributing workflow

Before committing changes:

1. run the affected chapter workflows;
2. inspect the generated results;
3. run `quarto render`;
4. inspect `docs/index.html` and the affected pages;
5. stage the source and generated files; and
6. run Git's whitespace check.

```bash
git add -A
git diff --cached --check
git status
```

Generated Quarto HTML under `docs/` is excluded from whitespace warnings through `.gitattributes`.

Commit and push after the checks pass:

```bash
git commit -m "docs: update machine-learning guide"
git push origin main
```

## Learning pathway

After completing this guide, continue with the related CDI guides:

- **Model Deployment** for serving, packaging, release strategies, and deployment operations;
- **From Models to Systems** for architecture, feedback loops, reliability, and system-level tradeoffs; and
- **Advanced Data Science** for deeper statistical, computational, and research-oriented methods.

## About Complex Data Insights

Complex Data Insights develops structured learning guides and practical projects across Data Science, Artificial Intelligence, Omics and Bioinformatics, Clinical and Medical Data, Career Development, and Research and Innovation.

This guide is fully available as an open learning resource.
