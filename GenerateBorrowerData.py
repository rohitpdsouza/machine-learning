import numpy as np
import pandas as pd

def generate_tight_clusters(n=10000, seed=123, out_path="borrower3.csv"):
    rng = np.random.default_rng(seed)

    # group sizes: Low 40%, Medium 25%, High 35%
    n_low = int(0.40 * n)
    n_med = int(0.25 * n)
    n_high = n - n_low - n_med

    # arrays
    customer_id = np.arange(1, n + 1)
    cluster_id = np.empty(n, dtype=object)
    income = np.empty(n, dtype=float)
    income_volatility = np.empty(n, dtype=float)
    credit_limit = np.empty(n, dtype=float)
    loan_amount = np.empty(n, dtype=float)
    transaction_frequency = rng.integers(20, 71, n)
    missed_payment_count = np.empty(n, dtype=int)
    age = np.empty(n, dtype=int)
    employment_type = np.empty(n, dtype=object)
    average_balance = np.empty(n, dtype=float)
    account_tenure = rng.integers(1, 16, n)
    region = rng.choice(["east", "west", "north", "south"], n)

    # Helper uniform
    def u(a, b, size):
        return rng.uniform(a, b, size)

    # Tight, non-overlapping ranges (discriminative features)
    # Income (strictly disjoint)
    #   High risk:    100,000   - 400,000
    #   Medium risk:  800,000   - 1,800,000
    #   Low risk:   2,500,000   - 5,000,000

    # income_volatility (disjoint)
    #   Low:   0.05 - 0.10
    #   Med:   0.11 - 0.20
    #   High:  0.30 - 0.50

    # credit_utilization (controlled via loan fraction of credit_limit)
    #   Low:   0.05 - 0.15
    #   Med:   0.25 - 0.45
    #   High:  0.70 - 0.95

    # Missed payments
    #   Low:   0
    #   Med:   1 - 6
    #   High:  7 - 20

    # Average balance (disjoint-ish)
    #   High risk:   10,000 - 200,000
    #   Medium risk: 200,001 - 1,000,000
    #   Low risk:    1,500,000 - 3,500,000

    # Fill Low
    idx_low = np.arange(0, n_low)
    cluster_id[idx_low] = "Low"
    income[idx_low] = u(2_500_000, 5_000_000, n_low).round(2)
    income_volatility[idx_low] = u(0.05, 0.10, n_low).round(4)
    # credit_limit per formula: credit_limit = (income_volatility * income * 2)
    credit_limit[idx_low] = (income_volatility[idx_low] * income[idx_low] * 2).round(0)
    # loan fraction to ensure low credit utilization
    loan_amount[idx_low] = (credit_limit[idx_low] * u(0.05, 0.15, n_low)).round(2)
    missed_payment_count[idx_low] = 0
    age[idx_low] = rng.integers(25, 60, n_low)
    employment_type[idx_low] = rng.choice(["salaried"], n_low)  # bias salaried
    average_balance[idx_low] = u(1_500_000, 3_500_000, n_low).round(2)

    # Fill Medium (split into two subtypes if needed)
    idx_med = np.arange(n_low, n_low + n_med)
    cluster_id[idx_med] = "Medium"
    # income
    income[idx_med] = u(800_000, 1_800_000, n_med).round(2)
    income_volatility[idx_med] = u(0.11, 0.20, n_med).round(4)
    credit_limit[idx_med] = (income_volatility[idx_med] * income[idx_med] * 2).round(0)
    # To satisfy both medium DTI/util and some with low DTI but high outstanding+age, split half/half
    half_med = n_med // 2
    a = idx_med[:half_med]  # medium DTI/util
    b = idx_med[half_med:]  # low DTI/util but high outstanding+age

    loan_amount[a] = (credit_limit[a] * u(0.25, 0.45, len(a))).round(2)  # medium utilization
    missed_payment_count[a] = rng.integers(1, 6, len(a))
    age[a] = rng.integers(30, 60, len(a))
    employment_type[a] = rng.choice(["contract", "salaried", "self-employed"], len(a))
    average_balance[a] = u(200_001, 1_000_000, len(a)).round(2)

    # b: low utilization but high outstanding & older
    loan_amount[b] = (credit_limit[b] * u(0.05, 0.18, len(b))).round(2)  # low utilization band
    missed_payment_count[b] = rng.integers(1, 5, len(b))
    age[b] = rng.integers(55, 70, len(b))
    employment_type[b] = rng.choice(["contract", "salaried"], len(b))
    average_balance[b] = u(250_000, 1_200_000, len(b)).round(2)

    # Fill High
    idx_high = np.arange(n_low + n_med, n)
    cluster_id[idx_high] = "High"
    income[idx_high] = u(100_000, 400_000, n_high).round(2)
    income_volatility[idx_high] = u(0.30, 0.50, n_high).round(4)
    credit_limit[idx_high] = (income_volatility[idx_high] * income[idx_high] * 2).round(0)
    # ensure minimum credit limit not too small
    credit_limit[idx_high] = np.maximum(credit_limit[idx_high], 50_000)
    loan_amount[idx_high] = (credit_limit[idx_high] * u(0.70, 0.95, n_high)).round(2)
    missed_payment_count[idx_high] = rng.integers(7, 21, n_high)
    age[idx_high] = rng.integers(40, 70, n_high)
    employment_type[idx_high] = rng.choice(["self-employed", "contract"], n_high)
    average_balance[idx_high] = u(10_000, 200_000, n_high).round(2)

    # Enforce loan_amount <= credit_limit (clip)
    loan_amount = np.minimum(loan_amount, credit_limit).round(2)

    # loan_amount_outstanding: risk-biased fractions
    loan_amount_outstanding = np.empty(n, dtype=float)
    loan_amount_outstanding[idx_low] = (loan_amount[idx_low] * u(0.0, 0.25, len(idx_low))).round(2)
    loan_amount_outstanding[a] = (loan_amount[a] * u(0.2, 0.5, len(a))).round(2)
    loan_amount_outstanding[b] = (loan_amount[b] * u(0.6, 0.98, len(b))).round(2)
    loan_amount_outstanding[idx_high] = (loan_amount[idx_high] * u(0.7, 1.0, len(idx_high))).round(2)
    loan_amount_outstanding = np.minimum(loan_amount_outstanding, loan_amount).round(2)

    # debt_to_income and credit_utilization (rounded)
    debt_to_income = np.round(loan_amount / np.clip(income, 1.0, None), 4)
    credit_utilization = np.round(loan_amount / np.clip(credit_limit, 1.0, None), 4)

    # transaction_frequency, account_tenure, region already set
    # Ensure employment_type filled where not set
    empty_emp = np.where(pd.isna(employment_type))[0]
    if len(empty_emp) > 0:
        employment_type[empty_emp] = rng.choice(["salaried", "self-employed", "contract"], len(empty_emp))

    # missed_payment_count array ensure ints
    missed_payment_count = missed_payment_count.astype(int)

    # Build DataFrame with required formatting
    df = pd.DataFrame({
        "cluster_id": cluster_id,
        "customer_id": customer_id,
        "income": income.round(2),
        "income_volatility": income_volatility.round(2),
        "credit_limit": credit_limit.astype(int),
        "loan_amount": loan_amount.round(2),
        "transaction_frequency": transaction_frequency,
        "missed_payment_count": missed_payment_count,
        "age": age.astype(int),
        "employment_type": employment_type,
        "average_balance": average_balance.round(2),
        "account_tenure": account_tenure.astype(int),
        "region": region,
        "debt_to_income": debt_to_income,
        "credit_utilization": credit_utilization,
        "loan_amount_outstanding": loan_amount_outstanding.round(2)
    })

    # Final sanity checks
    assert df["customer_id"].nunique() == n
    assert (df["loan_amount"] <= df["credit_limit"]).all()
    assert (df["loan_amount_outstanding"] <= df["loan_amount"]).all()

    # Save CSV
    df.to_csv(out_path, index=False)
    return df

if __name__ == "__main__":
    df = generate_tight_clusters()
    print("Saved borrower.csv with", len(df), "rows (Low/Med/High = 40%/25%/35%)")