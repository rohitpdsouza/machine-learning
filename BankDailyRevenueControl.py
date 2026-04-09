import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor

# -----------------------------
# STEP 1: Load the dataset
# -----------------------------

# Set pandas display options to avoid truncation
pd.set_option("display.max_columns", None)

# Define the file path
file_path = r"data\input\bank_revenue.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# -----------------------------
# STEP 2: Pre Processing
# -----------------------------

# Currency conversion: Define fixed FX rates to convert balances into USD. This is necessary because the dataset
# contains account balances in multiple currencies, and we need a common currency (USD) for training the models
# effectively.
fx_rates = {
    "EUR": 1.1,  # Euro to USD
    "GBP": 1.25,  # British Pound to USD
    "JPY": 0.0065,  # Japanese Yen to USD
    "USD": 1.0  # US Dollar to USD
}

# Map the account_currency to fx_rates and create a new column for the FX rate
df["fx_rate"] = df["account_currency"].map(fx_rates)
# Create a new column for account_balance in USD by multiplying the original balance with the FX rate
df["account_balance_usd"] = df["account_balance"] * df["fx_rate"]
df.drop(columns=["fx_rate"], inplace=True)

# Split the data into two separate DataFrames: one for revenue (overdrafts) and one for expenses (deposits). This is
# done to train the models separately for revenue and expense predictions, as they have different characteristics
df_revenue = df[df["account_balance_usd"] < 0].copy()
df_expense = df[df["account_balance_usd"] > 0].copy()

# Features used for training the models.
features = ["account_balance_usd", "client_rate", "days_in_year"]

# Scaling factor to magnify small interest values. Since the interest values can be very small (especially for
# deposits), we multiply them by a large factor (1 million) to make them more learnable for the machine learning
# model. This helps the model to capture patterns in the data more effectively.
scale = 1e6

# Training parameters for the XGBoost model. These parameters control the complexity and learning process of the model.
param_grid = {
    "n_estimators": [500, 1000, 2000, 5000, 10000],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [4, 6, 8, 10],
    "subsample": [0.5, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 8],
    "gamma": [0, 0.1, 0.2]
}

# Threshold for anomaly detection. After the model makes predictions, we will compare the predicted interest values
# with the actual values. If the absolute deviation is greater than this threshold (2 USD), we will flag it as an
# anomaly.
threshold = 1.5
# Percentage deviation threshold for larger values
percentage_threshold = 0.05   # 10%

# -----------------------------
# Step 3: Train revenue model
# -----------------------------

rev_features = df_revenue[features]
rev_interest = df_revenue["interest_revenue_usd"] * scale

# Perform a 20% train-test split
rev_train, rev_test, rev_interest_train, rev_interest_test = train_test_split(
    rev_features,
    rev_interest,
    test_size=0.2,
    random_state=42
)

# Define the XGBoost regressor

# RandomizedSearchCV is a hyperparameter tuning technique in scikit‑learn that allows
# you to search for the best combination of hyperparameters for a given model. It randomly samples a specified number
# of combinations from a defined hyperparameter space and evaluates the model's performance using cross-validation.
# This approach is more efficient than GridSearchCV when dealing with a large hyperparameter space, as it does not
# exhaustively evaluate all possible combinations but still provides a good chance of finding an optimal set of
# hyperparameters.
grid_rev = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=20,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=1,
    n_jobs=-1
)

# Runs the search and finds the best hyperparameter combination for the revenue model based on the negative mean
# squared error (MSE) metric. The best estimator is then stored in model_rev, and the best parameters are printed to
# the console. This process helps to optimize the performance of the revenue prediction model by finding the most
# effective hyperparameters.
grid_rev.fit(rev_train, rev_interest_train)
model_rev = grid_rev.best_estimator_
print("Best parameters for revenue model:", grid_rev.best_params_)

# Train the model
model_rev.fit(rev_train, rev_interest_train)

# Predict the interest revenue on the test set and scale back to USD
rev_interest_prediction = model_rev.predict(rev_test) / scale

# Scale back actual
rmse_revenue = np.sqrt(mean_squared_error(rev_interest_test / scale, rev_interest_prediction))

# Compute the Root Mean Squared Error (RMSE) to evaluate the accuracy of the revenue model. RMSE is a common metric
# for regression
print("Revenue Model RMSE on Test Set: $", round(rmse_revenue, 2), "\n")

# Predict interest revenue for the entire revenue dataset and scale back to USD
df_revenue["predicted_interest_revenue_usd"] = model_rev.predict(rev_features) / scale

# Compute the deviation between actual and predicted interest revenue, and flag anomalies where the absolute
# deviation exceeds the threshold
df_revenue["deviation"] = np.where(
    df_revenue["interest_revenue_usd"].abs() <= 50,
    df_revenue["interest_revenue_usd"] - df_revenue["predicted_interest_revenue_usd"],
    np.nan   # leave blank for larger values
)

df_revenue["percentage_deviation"] = np.where(
    df_revenue["interest_revenue_usd"].abs() > 50,
    (df_revenue["interest_revenue_usd"] - df_revenue["predicted_interest_revenue_usd"]).abs() /
    df_revenue["interest_revenue_usd"].abs(),
    np.nan   # leave blank for smaller values
)

# Compute the percentage deviation to identify significant anomalies relative to the actual value. This is important
# because an absolute deviation might be significant for small interest values but not for larger ones. By
# calculating the percentage deviation, we can flag anomalies that are significant in relative terms, even if the
# absolute deviation is small.
df_revenue["anomaly_flag"] = np.where(
    (df_revenue["interest_revenue_usd"].abs() <= 50) & (df_revenue["deviation"].abs() > threshold),
    "YES",
    np.where(
        (df_revenue["interest_revenue_usd"].abs() > 50) &
        (df_revenue["percentage_deviation"] > percentage_threshold),
        "YES",
        "NO"
    )
)

# -----------------------------
# Step 4: Train expense model
# -----------------------------

exp_features = df_expense[features]
exp_interest = df_expense["interest_expense_usd"] * scale

# Perform a 20% train-test split
exp_train, exp_test, exp_interest_train, exp_interest_test = train_test_split(
    exp_features,
    exp_interest,
    test_size=0.2,
    random_state=42
)

# Define the XGBoost regressor
grid_exp = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=20,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=1,
    n_jobs=-1
)

# Runs the search and finds the best hyperparameter combination for the expense model based on the negative mean
grid_exp.fit(exp_train, exp_interest_train)

# The best estimator is then stored in model_exp, and the best parameters are printed to the console. This process
# helps to optimize the performance of the expense prediction model by finding the most effective hyperparameters.
model_exp = grid_exp.best_estimator_
print("Best parameters for expense model:", grid_exp.best_params_)

# Train the model
model_exp.fit(exp_train, exp_interest_train)

# Predict the interest expense on the test set and scale back to USD
exp_interest_prediction = model_exp.predict(exp_test) / scale

# Compute the Root Mean Squared Error (RMSE) to evaluate the accuracy of the expense model
rmse_expense = np.sqrt(mean_squared_error(exp_interest_test / scale, exp_interest_prediction))
print("Expense Model RMSE on Test Set: $", round(rmse_expense, 2), "\n")

# Predict interest expense for the entire expense dataset and scale back to USD
df_expense["predicted_interest_expense_usd"] = model_exp.predict(exp_features) / scale

# Compute the deviation between actual and predicted interest expense, and flag anomalies where the absolute
# deviation exceeds the threshold
df_expense["deviation"] = np.where(
    df_expense["interest_expense_usd"].abs() <= 50,
    df_expense["interest_expense_usd"] - df_expense["predicted_interest_expense_usd"],
    np.nan   # leave blank for larger values
)

df_expense["percentage_deviation"] = np.where(
    df_expense["interest_expense_usd"].abs() > 50,
    (df_expense["interest_expense_usd"] - df_expense["predicted_interest_expense_usd"]).abs() /
    df_expense["interest_expense_usd"].abs(),
    np.nan   # leave blank for smaller values
)

df_expense["anomaly_flag"] = np.where(
    (df_expense["interest_expense_usd"].abs() <= 50) & (df_expense["deviation"].abs() > threshold),
    "YES",
    np.where(
        (df_expense["interest_expense_usd"].abs() > 50) &
        (df_expense["percentage_deviation"] > percentage_threshold),
        "YES",
        "NO"
    )
)

# -----------------------------
# Step 5: Combine and save output
# -----------------------------

# Combine df_expense and df_revenue into a single DataFrame
df_output = pd.concat([df_revenue, df_expense], axis=0)

# Add balance_type attribute to df_output
df_output["balance_type"] = np.where(df_output["account_balance"] > 0, "Deposit", "Overdraft")
# Flip the sign of deviation to reflect where prediction is more or less
df_output["deviation"] = np.where(
    df_output["deviation"].notna(),
    df_output["deviation"] * -1,
    np.nan
)

df_output["percentage_deviation"] = np.where(
    df_output["percentage_deviation"].notna(),
    (df_output["percentage_deviation"] * 100).round(2),
    np.nan
)

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
    "percentage_deviation",
    "anomaly_flag"
]
df_output = df_output[new_order]

# Print the count of each possible value of anomaly_flag
print(df_output["anomaly_flag"].value_counts())

# Calculate and print the total sums
total_interest_revenue_usd = df_output["interest_revenue_usd"].sum()
total_interest_expense_usd = df_output["interest_expense_usd"].sum()
total_predicted_interest_revenue_usd = df_output["predicted_interest_revenue_usd"].sum()
total_predicted_interest_expense_usd = df_output["predicted_interest_expense_usd"].sum()

print(f"\nTotal Actual Interest Revenue USD: {total_interest_revenue_usd}")
print(f"Total Actual Interest Expense USD: {total_interest_expense_usd}")
print(f"Total Predicted Interest Revenue USD: {total_predicted_interest_revenue_usd}")
print(f"Total Predicted Interest Expense USD: {total_predicted_interest_expense_usd}")

# Save the DataFrame to a CSV file (overwrites if the file already exists)
output_path = "data/output/bank_revenue_anomaly_detection.csv"
df_output.to_csv(output_path, index=False)
print(f"DataFrame saved to {output_path}")
