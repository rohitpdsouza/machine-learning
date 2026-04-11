# - Purpose: Automates detection of anomalies in bank revenue by training and applying ML models.
# - Data flow: Preprocesses bank_revenue.csv, scales small interest values, and trains revenue and expense models.
# - Output: Uses trained models to predict interest and flag anomalies, then writes results to a CSV for reporting.
# - Business value: Enables automated monitoring of revenue anomalies.


import numpy as np
import pandas as pd
import joblib
import yaml

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor

# Set pandas display options to avoid truncation
pd.set_option("display.max_columns", None)


def main() -> None:

    # -----------------------------
    # STEP 1: Pre Processing
    # -----------------------------
    # Define the revenue file path to be used for training the models
    file_path = r"data\input\bank_revenue.csv"
    # Load and pre-process the file into a dataframe
    df = pre_process(file_path)

    # Scaling factor to magnify small interest values. Since the interest values can be very small (especially for
    # deposits), we multiply them by a large factor (1 million) to make them more learnable for the machine learning
    # model. This helps the model to capture patterns in the data more effectively.
    scale = 1e6

    # -----------------------------
    # STEP 2: Train the revenue and expense models
    # -----------------------------
    training(df, scale)

    # -----------------------------
    # STEP 3: Load the trained models
    # -----------------------------
    revenue_model = load_model(model_path="revenue_model.pkl")
    expense_model = load_model(model_path="expense_model.pkl")

    # -----------------------------
    # STEP 4: Predict the interest  and anomalies using the trained models
    # -----------------------------
    file_path = r"data\input\bank_revenue.csv"
    # Load and pre-process the file into a dataframe
    df = pre_process(file_path)
    df_predicted = prediction(df, scale, revenue_model, expense_model)

    # -----------------------------
    # STEP 5: Reformat and save the output
    # -----------------------------
    output_path = "data/output/bank_revenue_anomaly_detection.csv"
    process_output(df_predicted, output_path)

    # -----------------------------
    # THE END!
    # -----------------------------


def process_output(df_output, output_path):
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

    # Save the DataFrame to a CSV file (overwrites if the file already exists)
    df_output.to_csv(output_path, index=False)
    print(f"\nDataFrame saved to {output_path}")

    return


def pre_process(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Currency conversion: Define fixed FX rates to convert balances into USD. This is necessary because the dataset
    # contains account balances in multiple currencies, and we need a common currency (USD) for training the models
    # effectively.
    fx_rate_path = r"data\input\usd_exchange_rates.csv"
    rates = pd.read_csv(
        fx_rate_path,
        usecols=["source_currency", "usd_exchange_rate"],
        dtype={"source_currency": str, "usd_exchange_rate": float}
    )
    rates.dropna(subset=["source_currency", "usd_exchange_rate"])
    # If duplicate currencies exist, keep the last occurrence
    rates = rates.drop_duplicates(subset="source_currency", keep="last")
    fx_rates = rates.set_index("source_currency")["usd_exchange_rate"].to_dict()

    # Map the account_currency to fx_rates and create a new column for the FX rate
    df["fx_rate"] = df["account_currency"].map(fx_rates)
    # Create a new column for account_balance in USD by multiplying the original balance with the FX rate
    df["account_balance_usd"] = df["account_balance"] * df["fx_rate"]
    df.drop(columns=["fx_rate"], inplace=True)
    return df


def training(df, scale):
    # -----------------------------
    # STEP 1: Access the training switches
    # -----------------------------
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    train_revenue = config["feature_flags"]["TRAIN_REVENUE"]
    train_expense = config["feature_flags"]["TRAIN_EXPENSE"]

    # -----------------------------
    # STEP 2: Revenue model
    # -----------------------------
    # Only train if training switch is on
    if train_revenue:
        # Split the revenue
        df_revenue = df[df["account_balance_usd"] < 0].copy()

        # Features used to train the model
        input_features = ["account_balance_usd", "client_rate", "days_in_year"]
        target_feature = "interest_revenue_usd"

        # Model save path
        model_path = "revenue_model.pkl"

        # Train the revenue model using the df_revenue DataFrame, the defined features, scaling factor, and target
        # feature
        print("\nTraining the revenue model...")
        train_model(df_revenue, input_features, scale, target_feature, model_path)
    else:
        print("Revenue model training is disabled. Previously trained model will be used for prediction if available.")

    # -----------------------------
    # STEP 3: Expense model
    # -----------------------------
    # Only train if training switch is on
    if train_expense:
        # Split the expense
        df_expense = df[df["account_balance_usd"] > 0].copy()

        # Features used to train the model
        input_features = ["account_balance_usd", "client_rate", "days_in_year"]
        target_feature = "interest_expense_usd"

        # Model save path
        model_path = "expense_model.pkl"

        # Train the expense model using the df_expense DataFrame, the defined features, scaling factor, and target
        # feature
        print("Training the expense model...")
        train_model(df_expense, input_features, scale, target_feature, model_path)
    else:
        print("Expense model training is disabled. Previously trained model will be used for prediction if available",
              "\n")

    return


def load_model(model_path):
    model = joblib.load(model_path)
    print(f"Model {model_path} loaded from the disk")
    return model


def prediction(df, scale, revenue_model, expense_model):
    # Threshold for anomaly detection. After the model makes predictions, we will compare the predicted interest values
    # with the actual values. If the absolute deviation is greater than this threshold (2 USD), we will flag it as an
    # anomaly.
    threshold = 1.5
    # Percentage deviation threshold for larger values
    percentage_threshold = 0.05  # 5%

    # -----------------------------
    # STEP 1: Predict the revenue
    # -----------------------------
    # Split the revenue
    df_revenue = df[df["account_balance_usd"] < 0].copy()

    # Features used to predict by the model
    input_features = ["account_balance_usd", "client_rate", "days_in_year"]
    input_df = df_revenue[input_features]
    print("Predicting the interest_revenue_usd and identifying anomalies...")
    df_revenue["predicted_interest_revenue_usd"] = revenue_model.predict(input_df) / scale

    # Compute the deviation between actual and predicted interest revenue, and flag anomalies where the absolute
    # deviation exceeds the threshold.
    # For small amounts <= 50 USD, we use absolute deviation, while for larger amounts
    # we use percentage deviation to account for the relative significance of the deviation. This approach allows us to
    # identify anomalies that are significant in both absolute and relative terms, depending on the size of the interest
    # revenue.
    df_revenue["deviation"] = np.where(
        df_revenue["interest_revenue_usd"].abs() <= 50,
        df_revenue["interest_revenue_usd"] - df_revenue["predicted_interest_revenue_usd"],
        np.nan  # leave blank for larger values
    )

    df_revenue["percentage_deviation"] = np.where(
        df_revenue["interest_revenue_usd"].abs() > 50,
        (df_revenue["interest_revenue_usd"] - df_revenue["predicted_interest_revenue_usd"]).abs() /
        df_revenue["interest_revenue_usd"].abs(),
        np.nan  # leave blank for smaller values
    )

    # Anomaly flag where deviation is greater than the threshold.
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
    # STEP 2: Predict the expense
    # -----------------------------
    # Split the expense
    df_expense = df[df["account_balance_usd"] > 0].copy()

    # Features used to predict by the model
    input_features = ["account_balance_usd", "client_rate", "days_in_year"]
    input_df = df_expense[input_features]
    print("Predicting the interest_expense_usd and identifying anomalies...")
    df_expense["predicted_interest_expense_usd"] = expense_model.predict(input_df) / scale

    # Compute the deviation between actual and predicted interest expense, and flag anomalies where the absolute
    # deviation exceeds the threshold
    df_expense["deviation"] = np.where(
        df_expense["interest_expense_usd"].abs() <= 50,
        df_expense["interest_expense_usd"] - df_expense["predicted_interest_expense_usd"],
        np.nan  # leave blank for larger values
    )

    df_expense["percentage_deviation"] = np.where(
        df_expense["interest_expense_usd"].abs() > 50,
        (df_expense["interest_expense_usd"] - df_expense["predicted_interest_expense_usd"]).abs() /
        df_expense["interest_expense_usd"].abs(),
        np.nan  # leave blank for smaller values
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
    # STEP 3: Combine df_expense and df_revenue into a single DataFrame and publish the prediction stats
    # -----------------------------
    df_output = pd.concat([df_revenue, df_expense], axis=0)

    # Print the count of each possible value of anomaly_flag
    print("\n", df_output["anomaly_flag"].value_counts())

    # Calculate and print the total sums
    total_interest_revenue_usd = df_output["interest_revenue_usd"].sum()
    total_interest_expense_usd = df_output["interest_expense_usd"].sum()
    total_predicted_interest_revenue_usd = df_output["predicted_interest_revenue_usd"].sum()
    total_predicted_interest_expense_usd = df_output["predicted_interest_expense_usd"].sum()

    print(f"\nTotal Actual Interest Revenue USD: {total_interest_revenue_usd}")
    print(f"Total Actual Interest Expense USD: {total_interest_expense_usd}")
    print(f"Total Predicted Interest Revenue USD: {total_predicted_interest_revenue_usd}")
    print(f"Total Predicted Interest Expense USD: {total_predicted_interest_expense_usd}")

    return df_output


def train_model(df, input_features, scale, target_feature, model_path):
    inputs = df[input_features]
    target = df[target_feature] * scale

    # Perform a 20% train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        inputs,
        target,
        test_size=0.2,
        random_state=42
    )

    # Define the grid model
    xgboost_grid = xgboost_def()

    # Runs the search and finds the best hyperparameter combination for the revenue model based on the negative mean
    # squared error (MSE) metric. The best estimator is then stored and the best parameters are printed to
    # the console. This process helps to optimize the performance of the  prediction model by finding the most
    # effective hyperparameters.
    xgboost_grid.fit(x_train, y_train)

    # The best estimator is then stored and the best parameters are printed to the console. This process
    # helps to optimize the performance of the prediction model by finding the most effective hyperparameters.
    xgboost_model = xgboost_grid.best_estimator_
    print("Best parameters for model:", xgboost_grid.best_params_)

    # Train the model
    xgboost_model.fit(x_train, y_train)

    # Predict the interest on the test set and scale back to USD
    y_test_predicted = xgboost_model.predict(x_test) / scale

    # Compute the Root Mean Squared Error (RMSE) to evaluate the accuracy of the expense model
    root_mean_squared_error = np.sqrt(mean_squared_error(y_test / scale, y_test_predicted))
    print("XGBoost Model RMSE on Test Set: $", round(root_mean_squared_error, 2))

    # Save model to disk
    joblib.dump(xgboost_model, model_path)
    print(f"Model {model_path} saved to disk", "\n")

    return


def xgboost_def():
    # Define the XGBoost regressor
    # RandomizedSearchCV is a hyperparameter tuning technique in scikit‑learn that allows you to search for the best
    # combination of hyperparameters for a given model. It randomly samples a specified number of combinations from a
    # defined hyperparameter space and evaluates the model's performance using cross-validation. This approach is
    # more efficient than GridSearchCV when dealing with a large hyperparameter space, as it does not exhaustively
    # evaluate all possible combinations but still provides a good chance of finding an optimal set of hyperparameters.
    param_grid = {
        "n_estimators": [500, 1000, 2000, 5000, 10000],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8, 10],
        "subsample": [0.5, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 8],
        "gamma": [0, 0.1, 0.2]
    }

    grid_model = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=42),
        param_distributions=param_grid,
        n_iter=20,
        scoring="neg_mean_squared_error",
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    return grid_model


if __name__ == "__main__":
    main()
