# CDI Git Hooks

This repository uses version-controlled Git hooks to automate routine maintenance tasks, including updating the guide date before a commit.

## Hook files

```text
.githooks/
├── install.sh
├── pre-commit
└── README.md
```

* `install.sh` configures the repository to use `.githooks/`.
* `pre-commit` runs automatically before each commit.
* `README.md` documents the setup and workflow.

## Initial setup

Run the installer once after cloning the repository or after adding the hooks to a new repository:

```bash
bash .githooks/install.sh
```

The installer makes the pre-commit hook executable and configures the hooks path for the current repository.

## Confirm the configuration

```bash
git config --local --get core.hooksPath
```

Expected output:

```text
.githooks
```

## Normal commit workflow

After the initial setup, use Git normally:

```bash
git add -A
git commit -m "docs: update guide content"
git push origin main
```

The pre-commit hook runs automatically during `git commit`. When meaningful guide source files have changed, it updates the `date` field in `_quarto.yml` and stages that update as part of the commit.

> **Note:** Git does not transfer local Git configuration when a repository is cloned. Run `.githooks/install.sh` once for every new clone.
