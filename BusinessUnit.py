"""
Use Case : We have a ledger in which each balance is attributed to a business unit of the bank. The business unit
generally depends on a combination of client and product attributes and these rules or mappings are generally setup
as reference data by the legal entity controllers. A missing rule or mapping in the reference data for a new product
or a new flavour of an existing product (e.g. new type of FX product) will attribute that balance to a default
business unit. The legal entity controllers monitor the default business unit daily for any suspense balances and
manually reclass them to the correct business unit.

This code will train the ML model on existing business unit reference data and will automatically predict the
business unit for balances where the rule or mapping is missing.
"""

# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# Prevent column width truncation
pd.set_option('display.max_colwidth', None)
# Optional: widen the display in console
pd.set_option('display.width', 1000)

# Step 2: Load your ledger data
df = pd.read_csv("data/input/trial_balance.csv")
print("\ntrial_balance\n")
print(df.to_string())

# Step 3: Separate rows with and without business_unit
train_df = df[df['business_unit'].notna() & df['business_unit'].str.strip().ne('')]
predict_df = df[df['business_unit'].isna() | df['business_unit'].str.strip().eq('')]
print("\ntrain_test_df\n", train_df.to_string())
print("\npredict_df\n", predict_df.to_string())

# Step 4: Select feature and target columns
features = ['product_type', 'product_sub_type', 'portfolio_account_type', 'credit_purpose_code', 'investor_location']

X = train_df[features].copy()
y = train_df['business_unit']

# Step 5: Encode categorical text data into numbers using one-hot encoding
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ],
    remainder='passthrough'
)

# Step 6: Build the model pipeline
rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model_pipeline = Pipeline(steps=[
    ('encoder', encoder),
    ('classifier', rf_model)
])

# Step 7: Split data for testing and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nX_train:\n", X_train)
print("\ny_train:\n", y_train)
print("\nX_test:\n", X_test)
print("\ny_test:\n", y_test)

# Step 8: Train the random forest model
model_pipeline.fit(X_train, y_train)

# Step 9: Evaluate model accuracy
y_pred = model_pipeline.predict(X_test)

print("\ny_pred:\n", y_pred)
print("\nClassification Report\n", classification_report(y_test, y_pred))

# Step 10: Predict missing business_unit values
X_pred = predict_df[features].copy()
predict_df = predict_df.copy()
predict_df['business_unit'] = model_pipeline.predict(X_pred)
print("\npredict_df\n", predict_df.to_string())

# Step 11: Combine back into a single dataset
final_df = pd.concat([train_df, predict_df])

print("\nFinal\n", final_df)
