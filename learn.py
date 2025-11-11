# ------------------------------------------------------------
# Business Unit Prediction with Random Forest (Teaching Version)
# ------------------------------------------------------------

# Step 1: Import required libraries
# "import" brings in external packages so we can use their functions/classes
import pandas as pd                     # pandas: for working with tabular data (DataFrame, Series)
import numpy as np                      # numpy: for numerical arrays and math operations

# scikit-learn (sklearn) is a machine learning library
from sklearn.model_selection import train_test_split   # splits data into train/test sets
from sklearn.preprocessing import OneHotEncoder        # encodes categorical text into numbers
from sklearn.ensemble import RandomForestClassifier    # ML model: ensemble of decision trees
from sklearn.metrics import classification_report      # prints precision/recall/F1 metrics
from sklearn.compose import ColumnTransformer          # applies different preprocessing to different columns
from sklearn.pipeline import Pipeline                  # chains preprocessing + model into one object

# ------------------------------------------------------------
# Step 2: Configurable parameter
# ------------------------------------------------------------
top_n = 3  # integer: how many top business units to suggest (can change to 2, 3, etc.)

# ------------------------------------------------------------
# Step 3: Pandas display settings (for printing DataFrames nicely)
# ------------------------------------------------------------
pd.set_option('display.max_rows', None)       # show all rows
pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.max_colwidth', None)  # don't truncate long text
pd.set_option('display.width', 10000)        # widen console display

# ------------------------------------------------------------
# Step 4: Load your ledger data
# ------------------------------------------------------------
df = pd.read_csv("data/input/trial_balance.csv")   # reads CSV file into a pandas DataFrame
print("\ntrial_balance\n")
print(df.to_string())                   # .to_string() prints the whole DataFrame

# ------------------------------------------------------------
# Step 5: Separate rows with and without business_unit
# ------------------------------------------------------------
# Boolean indexing: we filter rows based on conditions
train_df = df[df['business_unit'].notna() & df['business_unit'].str.strip().ne('')]
# .notna() → True if not missing
# .str.strip() → remove spaces
# .ne('') → not equal to empty string
# & → logical AND

predict_df = df[df['business_unit'].isna() | df['business_unit'].str.strip().eq('')]
# .isna() → True if missing
# .eq('') → equal to empty string
# | → logical OR

print("\ntrain_test_df\n", train_df.to_string())
print("\npredict_df\n", predict_df.to_string())

# ------------------------------------------------------------
# Step 6: Select feature and target columns
# ------------------------------------------------------------
features = ['product_type', 'product_sub_type', 'portfolio_account_type',
            'credit_purpose_code', 'investor_location']  # list of strings

X = train_df[features].copy()          # DataFrame of input features
y = train_df['business_unit']          # Series of target labels

# ------------------------------------------------------------
# Step 7: Encode categorical text data into numbers
# ------------------------------------------------------------
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
# .select_dtypes → selects columns by type
# .columns → column names
# .tolist() → convert to Python list

numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

# ColumnTransformer lets us apply different preprocessing to different column groups
encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),  # encode categorical
        ('num', 'passthrough', numerical_cols)                             # keep numeric as is
    ],
    remainder='passthrough'  # keep any other columns unchanged
)

# ------------------------------------------------------------
# Step 8: Build the model pipeline
# ------------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,        # number of trees
    class_weight='balanced', # handle class imbalance
    random_state=42          # reproducibility
)

# Pipeline chains preprocessing (encoder) + model (classifier)
model_pipeline = Pipeline(steps=[
    ('encoder', encoder),
    ('classifier', rf_model)
])

# ------------------------------------------------------------
# Step 9: Split data for training and testing
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% test data
    random_state=42      # reproducibility
)

print("\nX_train:\n", X_train)
print("\ny_train:\n", y_train)
print("\nX_test:\n", X_test)
print("\ny_test:\n", y_test)

# ------------------------------------------------------------
# Step 10: Train the model
# ------------------------------------------------------------
model_pipeline.fit(X_train, y_train)   # .fit() trains the pipeline

# ------------------------------------------------------------
# Step 11: Evaluate model accuracy
# ------------------------------------------------------------
y_pred = model_pipeline.predict(X_test)   # .predict() makes predictions
print("\ny_pred:\n", y_pred)
print("\nClassification Report\n", classification_report(y_test, y_pred))

# ------------------------------------------------------------
# Step 12: Predict missing business_unit with top-N suggestions
# ------------------------------------------------------------
if not predict_df.empty:
    X_pred = predict_df[features].copy()

    # Get prediction probabilities
    classifier = model_pipeline.named_steps["classifier"]  # access model inside pipeline
    class_labels = classifier.classes_                     # list of class names
    class_probs = model_pipeline.predict_proba(X_pred)     # 2D array: (rows, classes)

    print("\nclass_labels:\n", class_labels)
    print("\nclass_probs:\n", class_probs)

    # Sort probabilities for each row
    sorted_indices = np.argsort(class_probs, axis=1)       # ascending order indices
    descending_indices = np.flip(sorted_indices, axis=1)   # flip to descending
    top_n_indices = descending_indices[:, :top_n]          # take top-N
    print("\ntop_n_indices\n", top_n_indices)

    # Collect top predictions into a list of dictionaries
    top_predictions: list[dict[str, object]] = []   # type hint: list of dicts
    for probs_row in class_probs:
        s = pd.Series(probs_row, index=class_labels)   # Series: probabilities with labels
        top = s.nlargest(top_n)                        # top-N largest values
        # Build dictionary of top labels and probabilities
        row_result = {f"top{i + 1}": label for i, label in enumerate(top.index)}
        row_result.update({
            f"probability_top{i + 1}": round(top[label] * 100, 4)
            for i, label in enumerate(top.index)
        })
        top_predictions.append(row_result)

    print("\n", top_predictions)

    # Convert list of dicts into DataFrame
    top_df = pd.DataFrame(top_predictions, index=predict_df.index)

    # Merge predictions back into original DataFrame
    predict_df = pd.concat([predict_df, top_df], axis=1)

    # Assign top1 as the chosen business_unit
    predict_df["business_unit"] = predict_df["top1"]

    # Display final result
    display_cols = (
        features
        + ["business_unit"]
        + [f"top{i}" for i in range(1, top_n + 1)]
        + [f"probability_top{i}" for i in range(1, top_n + 1)]
    )

    print(f"\nPredicted missing business units with top-{top_n} suggestions:\n")
    print(predict_df[display_cols].to_string(index=False))