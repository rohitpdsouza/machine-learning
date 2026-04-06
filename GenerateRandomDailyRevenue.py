import pandas as pd
import numpy as np

# -----------------------------
# Parameters
# -----------------------------
n_records = 50000
business_date = "20260406"  # same date for all records

# Example FX rates (replace with actual if needed)
fx_rates = {
    "USD": 1.0,
    "GBP": 1.25,
    "EUR": 1.1,
    "JPY": 0.0065
}
currencies = list(fx_rates.keys())

# -----------------------------
# Generate synthetic dataset
# -----------------------------
np.random.seed(42)

# Account numbers: 7-digit, cannot start with 0
account_numbers = np.random.randint(1000000, 9999999, size=n_records).astype(str)

# Random currency assignment
account_currency = np.random.choice(currencies, size=n_records)

# Account balances (can be positive or negative, wide range)
account_balance = np.round(np.random.uniform(-1e6, 1e6, size=n_records), 2)

# Client rates (precision 11,5 → realistic small percentages)
client_rate = np.round(np.random.uniform(0.0, 10.0, size=n_records), 5)

# Days in year depending on currency (360 or 365)
days_in_year = np.where(account_currency == "USD", 365, 360)

# Convert balances to USD
account_balance_usd = [
    bal * fx_rates[cur] for bal, cur in zip(account_balance, account_currency)
]

# Interest calculation (bank’s view)
interest_usd = []
interest_revenue_usd = []
interest_expense_usd = []

for bal_usd, rate, days in zip(account_balance_usd, client_rate, days_in_year):
    daily_interest = bal_usd * ((rate / 100) / days)
    if bal_usd < 0:
        # overdraft → bank earns revenue (positive)
        interest_revenue_usd.append(abs(daily_interest))
        interest_expense_usd.append(0.0)
        interest_usd.append(abs(daily_interest))
    else:
        # deposit → bank pays expense (negative)
        interest_revenue_usd.append(0.0)
        interest_expense_usd.append(-abs(daily_interest))
        interest_usd.append(-abs(daily_interest))

# -----------------------------
# Assemble DataFrame
# -----------------------------
df = pd.DataFrame({
    "business_date": business_date,
    "account_number": account_numbers,
    "account_currency": account_currency,
    "account_balance": account_balance,
    "client_rate": client_rate,
    "days_in_year": days_in_year,
    "interest_revenue_usd": interest_revenue_usd,
    "interest_expense_usd": interest_expense_usd
})

# -----------------------------
# Save to CSV
# -----------------------------
df.to_csv("C:/Users/prohi/PycharmProjects/POC/ML/data/output/bank_revenue.csv", index=False)

print("Synthetic dataset generated: bank_revenue.csv with", len(df), "records")
print(df.head())