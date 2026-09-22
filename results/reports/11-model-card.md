# Model Card: Customer Retention Support Model

## Purpose
Prioritize customers for voluntary retention-support review. Model version: 1.0.0.

## Prohibited uses
Do not deny service, change prices, or take adverse action automatically.

## Evaluation
Chronological final-month holdout with 253 rows. ROC AUC: 0.739. Decision threshold: 0.55.

## Feature policy
Uses tenure_months, monthly_charge, support_contacts, usage_hours, plan, region. `access_group` is retained for approved auditing only and excluded from model features.

## Human oversight
Every action requires review. Reviewers may override, must record reasons, and must provide an appeal route.

## Limitations
Synthetic educational data; group estimates vary with sample size; predictive associations are not causal; results are not approval for real-world use.

## Monitoring
Monitor input quality, drift, mature-label performance, group metrics, overrides, complaints, and incidents. Pause use when stop conditions are met.
