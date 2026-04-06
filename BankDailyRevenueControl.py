import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import os

# -----------------------------
# Step 1: Load dataset
# -----------------------------
df = pd.read_csv("data/input/bank_revenue.csv")

# -----------------------------
# Step 2: Preprocessing
# -----------------------------
fx_rates = {"USD": 1.0, "GBP": 1.25, "EUR": 1.1, "JPY": 0.0065}

df["account_balance_usd"] = df.apply(
    lambda row: row["account_balance"] * fx_rates.get(row["account_currency"], 1.0),
    axis=1
)
df["balance_type"] = df["account_balance_usd"].apply(lambda x: 1 if x > 0 else -1)

# Features
features = ["account_balance_usd", "client_rate", "days_in_year"]

# Scaling factor
SCALE = 1e6

# Training parameters
n_estimators = 10000
max_depth = 10
learning_rate = 0.01

# threshold parameter
threshold = 2

# -----------------------------
# Step 3: Split into revenue vs expense datasets
# -----------------------------
df_revenue = df[df["account_balance_usd"] < 0].copy()
df_expense = df[df["account_balance_usd"] > 0].copy()

# -----------------------------
# Step 4: Train revenue model
# -----------------------------

X_rev = df_revenue[features]
y_rev = df_revenue["interest_revenue_usd"] * SCALE  # scale target

X_train_rev, X_test_rev, y_train_rev, y_test_rev = train_test_split(X_rev, y_rev, test_size=0.2, random_state=42)

model_rev = XGBRegressor(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
model_rev.fit(X_train_rev, y_train_rev)

y_pred_rev = model_rev.predict(X_test_rev) / SCALE  # scale back
rmse_rev = np.sqrt(mean_squared_error(y_test_rev / SCALE, y_pred_rev))
print(f"Revenue Model RMSE (scaled back): {rmse_rev:.6f}")

df_revenue["predicted_interest_revenue_usd"] = model_rev.predict(X_rev) / SCALE
df_revenue["deviation"] = df_revenue["interest_revenue_usd"] - df_revenue["predicted_interest_revenue_usd"]
df_revenue["anomaly_flag"] = np.where(df_revenue["deviation"] > threshold, "YES", "NO")

# -----------------------------
# Step 5: Train expense model
# -----------------------------
X_exp = df_expense[features]
y_exp = df_expense["interest_expense_usd"] * SCALE  # scale target

X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(X_exp, y_exp, test_size=0.2, random_state=42)

model_exp = XGBRegressor(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
model_exp.fit(X_train_exp, y_train_exp)

y_pred_exp = model_exp.predict(X_test_exp) / SCALE
rmse_exp = np.sqrt(mean_squared_error(y_test_exp / SCALE, y_pred_exp))
print(f"Expense Model RMSE: {rmse_exp:.6f}")

df_expense["predicted_interest_expense_usd"] = model_exp.predict(X_exp) / SCALE
df_expense["deviation"] = df_expense["interest_expense_usd"] - df_expense["predicted_interest_expense_usd"]
df_expense["anomaly_flag"] = np.where(df_expense["deviation"].abs() > threshold, "YES", "NO")

# -----------------------------
# Step 6: Combine and save output
# -----------------------------
df_output = pd.concat([df_revenue, df_expense], axis=0)
df_output["balance_type"] = df_output["balance_type"].map({-1: "Overdraft", 1: "Deposit"})
# Define the desired column order
new_order = [
    "business_date",
    "account_number",
    "account_currency",
    "account_balance",
    "account_balance_usd",
    "balance_type",
    "client_rate",
    "days_in_year",
    "interest_revenue_usd",
    "interest_expense_usd",
    "predicted_interest_revenue_usd",
    "predicted_interest_expense_usd",
    "deviation",
    "anomaly_flag"
]
# Reorder the DataFrame
df_output = df_output[new_order]

output_path = "data/output/bank_revenue_anomaly_detection.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_output.to_csv(output_path, index=False)

print(f"Output saved to {output_path}")
print(df_output.head())
