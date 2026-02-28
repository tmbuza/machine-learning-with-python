import numpy as np
import pandas as pd

rng = np.random.default_rng(123)

n = 800

contract_type = rng.choice(["month-to-month", "one-year", "two-year"], size=n, p=[0.55, 0.30, 0.15])
tenure_months = rng.integers(1, 73, size=n)
monthly_spend = rng.normal(65, 18, size=n).clip(15, 160)
support_calls = rng.poisson(1.2, size=n).clip(0, 10)
autopay = rng.choice(["yes", "no"], size=n, p=[0.62, 0.38])

# Risk score (higher means more likely churn)
risk = (
    0.025 * (monthly_spend - 60)
    - 0.015 * (tenure_months - 24)
    + 0.22 * support_calls
    + np.where(contract_type == "month-to-month", 0.65, 0.0)
    + np.where(contract_type == "two-year", -0.35, 0.0)
    + np.where(autopay == "no", 0.25, 0.0)
    + rng.normal(0, 0.35, size=n)
)

p = 1 / (1 + np.exp(-risk))
churn = rng.binomial(1, p, size=n)

df = pd.DataFrame({
    "customer_id": [f"C{100000+i}" for i in range(n)],
    "tenure_months": tenure_months,
    "monthly_spend": np.round(monthly_spend, 2),
    "support_calls": support_calls,
    "contract_type": contract_type,
    "autopay": autopay,
    "churn": churn
})

out = "data/ml-ready/cdi-customer-churn.csv"
df.to_csv(out, index=False)
print(f"Wrote {out}")
