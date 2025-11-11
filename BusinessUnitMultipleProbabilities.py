"""
Use Case : We have a ledger in which each balance is attributed to a business unit of the bank. The business unit
generally depends on a combination of client and product attributes and these rules or mappings are generally setup
as reference data by the legal entity controllers. A missing rule or mapping in the reference data for a new product
or a new flavour of an existing product (e.g. new type of FX product) will attribute that balance to a default
business unit. The legal entity controllers monitor the default business unit daily for any suspense balances and
manually reclass them to the correct business unit.

This code will train the ML model on existing business unit reference data and will make top 3 possible business unit
suggestions along with probabilities.
"""

# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Configurable parameter: how many top business units to suggest
top_n = 3  # change to 2, 3, etc.

# Show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# Prevent column width truncation
pd.set_option('display.max_colwidth', None)
# Optional: widen the display in console
pd.set_option('display.width', 10000)

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

# Step 10: Predict missing business_unit with top-N suggestions
if not predict_df.empty:
    X_pred = predict_df[features].copy()
    predict_df = predict_df.copy()

    # Get prediction probabilities
    classifier = model_pipeline.named_steps["classifier"]
    # Get the class labels learned from training the model using .fit()
    class_labels = classifier.classes_
    class_probs = model_pipeline.predict_proba(X_pred)  # shape: (n_samples, n_classes)
    print("\nclass_labels:\n", class_labels)
    print("\nclass_probs:\n", class_probs)

    # Sort probabilities across each row (i.e., per record)
    sorted_indices = np.argsort(class_probs, axis=1)
    # Reverse order to get descending (highest first)
    descending_indices = np.flip(sorted_indices, axis=1)
    # Get top N
    top_n_indices = descending_indices[:, :top_n]
    print("\ntop_n_indices\n", top_n_indices)

    # Prepare empty lists for results
    top_predictions: list[dict[str, object]] = []
    # Iterate over each prediction row
    for probs_row in class_probs:
        # Convert row to Series with labels as index
        s = pd.Series(probs_row, index=class_labels)
        # Get top-N predictions
        top = s.nlargest(top_n)
        # Build result dict in the same format as before
        row_result = {
            f"top{i + 1}": label
            for i, label in enumerate(top.index)
        }
        row_result.update({
            f"probability_top{i + 1}": round(top[label] * 100, 4)
            for i, label in enumerate(top.index)
        })
        top_predictions.append(row_result)

    print("\n", top_predictions)

    # Convert to DataFrame and merge back
    top_df = pd.DataFrame(top_predictions, index=predict_df.index)

    # Merge with prediction DataFrame
    predict_df = pd.concat([predict_df, top_df], axis=1)

    # Assign the best (top1) prediction as the chosen business_unit
    predict_df["business_unit"] = predict_df["top1"]

    # Display result
    display_cols = (
        features
        + ["business_unit"]
        + [f"top{i}" for i in range(1, top_n + 1)]
        + [f"probability_top{i}" for i in range(1, top_n + 1)]
    )

    print(f"\nPredicted missing business units with top-{top_n} suggestions:\n")
    print(predict_df[display_cols].to_string(index=False))
