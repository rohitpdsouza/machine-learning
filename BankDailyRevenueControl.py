import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import os

# -----------------------------
# Step 1: Load dataset
# -----------------------------

# Read the input CSV file containing account balances, interest values, and other features
df = pd.read_csv("data/input/bank_revenue.csv")

# -----------------------------
# Step 2: Preprocessing
# -----------------------------

# Define fixed FX rates to convert balances into USD
fx_rates = {"USD": 1.0, "GBP": 1.25, "EUR": 1.1, "JPY": 0.0065}

# Convert account balances into USD using FX rates
df["account_balance_usd"] = df.apply(
    lambda row: row["account_balance"] * fx_rates.get(row["account_currency"], 1.0),
    axis=1
)

# Create balance_type: 1 for deposits (positive balances), -1 for overdrafts (negative balances)
df["balance_type"] = df["account_balance_usd"].apply(lambda x: 1 if x > 0 else -1)

# Define features used for training (balance_type excluded since models are trained separately)
features = ["account_balance_usd", "client_rate", "days_in_year"]

# Scaling factor to magnify very small interest values so the model can learn better
SCALE = 1e6

# Training parameters for XGBoost
n_estimators = 10000
max_depth = 10
learning_rate = 0.01

# Threshold for anomaly detection (absolute deviation > 2 USD is flagged)
threshold = 2

# -----------------------------
# Step 3: Split into revenue vs expense datasets
# -----------------------------

# Revenue dataset: overdrafts (negative balances)
df_revenue = df[df["account_balance_usd"] < 0].copy()
# Expense dataset: deposits (positive balances)
df_expense = df[df["account_balance_usd"] > 0].copy()

# -----------------------------
# Step 4: Train revenue model
# -----------------------------

# Features and scaled target for revenue
X_rev = df_revenue[features]
y_rev = df_revenue["interest_revenue_usd"] * SCALE  # scale target

# Train/test split
X_train_rev, X_test_rev, y_train_rev, y_test_rev = train_test_split(X_rev, y_rev, test_size=0.2, random_state=42)

# Define and train XGBoost regressor
model_rev = XGBRegressor(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
model_rev.fit(X_train_rev, y_train_rev)

# Predict and scale back to USD
y_pred_rev = model_rev.predict(X_test_rev) / SCALE  # scale back

# Compute RMSE for model accuracy
rmse_rev = np.sqrt(mean_squared_error(y_test_rev / SCALE, y_pred_rev))
print(f"Revenue Model RMSE (scaled back): {rmse_rev:.6f}")

# Store predictions, compute deviation, and flag anomalies
df_revenue["predicted_interest_revenue_usd"] = model_rev.predict(X_rev) / SCALE
df_revenue["deviation"] = df_revenue["interest_revenue_usd"] - df_revenue["predicted_interest_revenue_usd"]
df_revenue["anomaly_flag"] = np.where(df_revenue["deviation"] > threshold, "YES", "NO")

# -----------------------------
# Step 5: Train expense model
# -----------------------------

# Features and scaled target for expense
X_exp = df_expense[features]
y_exp = df_expense["interest_expense_usd"] * SCALE  # scale target

# Train/test split
X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(X_exp, y_exp, test_size=0.2, random_state=42)

# Define and train XGBoost regressor
model_exp = XGBRegressor(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
model_exp.fit(X_train_exp, y_train_exp)

# Predict and scale back to USD
y_pred_exp = model_exp.predict(X_test_exp) / SCALE

# Compute RMSE for model accuracy
rmse_exp = np.sqrt(mean_squared_error(y_test_exp / SCALE, y_pred_exp))
print(f"Expense Model RMSE: {rmse_exp:.6f}")

# Store predictions, compute deviation, and flag anomalies
df_expense["predicted_interest_expense_usd"] = model_exp.predict(X_exp) / SCALE
df_expense["deviation"] = df_expense["interest_expense_usd"] - df_expense["predicted_interest_expense_usd"]
df_expense["anomaly_flag"] = np.where(df_expense["deviation"].abs() > threshold, "YES", "NO")

# -----------------------------
# Step 6: Combine and save output
# -----------------------------

# Combine revenue and expense results
df_output = pd.concat([df_revenue, df_expense], axis=0)

# Replace balance_type numeric values with descriptive labels
df_output["balance_type"] = df_output["balance_type"].map({-1: "Overdraft", 1: "Deposit"})

# Define the desired column order for readability
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
df_output = df_output[new_order]

# Save final output to CSV
output_path = "data/output/bank_revenue_anomaly_detection.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_output.to_csv(output_path, index=False)

print(f"Output saved to {output_path}")
print(df_output.head())
