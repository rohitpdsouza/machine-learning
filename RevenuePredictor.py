"""
Use Case : Train a non-linear regression model to predict revenue for the next 3 months.
"""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

pd.set_option('display.max_columns', None)
# Prevent column width truncation
pd.set_option('display.max_colwidth', None)
# Optional: widen the display in console
pd.set_option('display.width', 1000)

# --------------------------
# Step 1: Read CSV
# --------------------------
df = pd.read_csv("data/input/monthly_revenue.csv")

# Quick view
print("Rows read:", len(df))
print(df.head())

# --------------------------
# Step 2: Parse period into datetime and extract time features
# --------------------------
# Assume period strings like 'Jan-25' or 'Jan-2025' — try month-year parse
# If format differs, adjust format string accordingly.
def parse_period(p):
    # Try formats common in examples: 'Jan-25' -> day is end of month
    try:
        return pd.to_datetime(p, format="%b-%y")
    except Exception:
        try:
            return pd.to_datetime(p, format="%b-%Y")
        except Exception:
            return pd.to_datetime(p)  # fallback

df["period_dt"] = df["period"].apply(parse_period)
# Use month start to be consistent
df["period_dt"] = df["period_dt"].dt.to_period("M").dt.to_timestamp("M")

# Extract numeric time features
df["year"] = df["period_dt"].dt.year
df["month"] = df["period_dt"].dt.month

print("\n",(df))

# --------------------------
# Step 3: Prepare feature matrix X and target y
# --------------------------
# We'll predict net_revenue_usd at product-month level
feature_cols = ["product", "year", "month"]
target_col = "net_revenue_usd"

X = df[feature_cols].copy()
y = df[target_col].copy()


# --------------------------
# Step 4: Encode categorical feature columsn for pipeline
# - OneHotEncode 'product'
# - pass through year and month (numeric)
# --------------------------
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

print("\nCategorical feature columns:\n", categorical_cols)
print("\nNumerical feature columns:\n", numerical_cols)

encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ],
    remainder='passthrough'
)

# --------------------------
# Step 5: Build preprocessing + model pipeline
# --------------------------

rfr_model = RandomForestRegressor(n_estimators=200, random_state=42)
model_pipeline = Pipeline(steps=[
    ('encoder', encoder),
    ('regressor', rfr_model)
])

# --------------------------
# Step 5: Quick evaluation (train/test split)
# Note: For time series it's better to use time-based split--to be done later
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# Step 6: Train the regressor model
# --------------------------
model_pipeline.fit(X_train, y_train)

# --------------------------
# Step 7: Evaluate the model
# --------------------------
y_pred = model_pipeline.predict(X_test)
print("\nEvaluation on test split:")
print("\nMAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("\nR2 :", round(r2_score(y_test, y_pred), 4))

#inspect a few actual vs predicted
compare = X_test.copy()
compare["actual"] = y_test.values
compare["predicted"] = y_pred
print("\nSample predictions (test set):\n")
print(compare.head(8).to_string(index=False))

# --------------------------
# Step 8: Prepare future periods to predict (next 3 months after max period in CSV)
# --------------------------
last_period = df["period_dt"].max()
next_start = last_period + pd.DateOffset(months=1)
future_periods_dt = pd.date_range(start=next_start, periods=3, freq="ME") #Month-end
print("\n Predict future periods :", future_periods_dt)

# All unique products observed in data
products = sorted(df["product"].unique())

future_rows = []
for dt in future_periods_dt:
    for prod in products:
        future_rows.append({
            "period_dt": dt,
            "period": dt.strftime("%b-%y"),
            "product": prod,
            "year": dt.year,
            "month": dt.month
        })

future_df = pd.DataFrame(future_rows, columns=["period", "product", "period_dt", "year", "month"])
#print(future_df)

# --------------------------
# Step 9: Predict net_revenue_usd for future_df
# --------------------------
X_future = future_df[feature_cols]
future_df["predicted_net_revenue_usd"] = model_pipeline.predict(X_future)

# --------------------------
# Step 10: Present results (print to console)
# --------------------------
print("\nPredictions for next 3 months (per product):\n")
out = future_df[["period", "product", "predicted_net_revenue_usd"]].copy()
out["predicted_net_revenue_usd"] = out["predicted_net_revenue_usd"].round(2)
print(out.to_string(index=False))