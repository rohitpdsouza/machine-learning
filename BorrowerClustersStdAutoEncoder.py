"""
A standard autoencoder is a type of neural network that learns to compress borrower data (like income,
credit limit, and missed payments) into a smaller set of numbers, and then reconstruct it — helping the model
discover key patterns that best represent each borrower.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans

# Show all columns
pd.set_option('display.max_columns', None)
# Prevent column width truncation
pd.set_option('display.max_colwidth', None)
# Widen the display in console
pd.set_option('display.width', None)
# Show full array (no '...') and prevent line wrapping
np.set_printoptions(edgeitems=3, linewidth=np.inf, suppress=True)

# Show all rows and columns
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
# Prevent column width truncation
pd.set_option('display.max_colwidth', None)
# Optional: widen the display in console
pd.set_option('display.width', None)

# ============================================
# Main Workflow
# ============================================


def main() -> None:
    # STEP 1: Generate borrower data
    # -----------------------------------------------------------------
    # borrower_df = generate_borrower_data(n_samples=10000)
    borrower_df = pd.read_csv("data/input/borrower2.csv", header=0)
    print("\nborrower_df\n", borrower_df.shape)

    # STEP 2: Encode categorical features and scale all features equally
    # -----------------------------------------------------------------
    borrower_df_encoded, borrower_scaled = pre_process(borrower_df)
    print("\nborrower_df encoded\n", borrower_df_encoded.head(5))
    x_scaled_df = pd.DataFrame(borrower_scaled, columns=borrower_df_encoded.columns.tolist())
    print("\nborrower_df scaled\n", x_scaled_df.head(5))

    print("\nborrower_df encoded shape:\n", borrower_df_encoded.shape)
    print("\nborrower_df scaled shape:\n", x_scaled_df.shape)

    # STEP 3: Build autoencoder to compress and de-compress the input and output
    # -----------------------------------------------------------------
    # If input_dim < 10, try encoding_dim in [2, 5, min(8, input_dim-1)].
    # If 10 ≤ input_dim ≤ 100, try encoding_dim ≈ 5–20% of input_dim (round to integer).
    # If input_dim > 100, try encoding_dim ≈ 2–10% of input_dim and consider stronger regularization.
    # If you have domain knowledge about latent factors, set encoding_dim close to that number.

    input_dim = borrower_scaled.shape[1]
    encoding_dim = 6  # latent space size
    print("\nNo of input features: ", input_dim, "\n")
    print("\nNo of latent features: ", encoding_dim , "\n")
    autoencoder, encoder, decoder = build_autoencoder(input_dim, encoding_dim)

    # STEP 4 : Compile the auto-encoder
    # -----------------------------------------------------------------
    # Use the Adam optimization algorithm to minimize mean squared error between the inputs and the
    # reconstructed outputs
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    autoencoder.compile(opt, loss='mse')
    #autoencoder.compile(opt, loss='mae')

    # STEP 5 : Train autoencoder
    # -----------------------------------------------------------------
    # The model tries to minimize the difference between input and reconstructed output and
    # automatically learns the feature weights in the process
    epochs = 200
    batch_size = 64
    history = train_autoencoder(borrower_scaled, autoencoder, epochs, batch_size)
    print(f"Autoencoder trained for {len(history.epoch)} epochs. Final reconstruction loss: {history.history['loss'][-1]:.4f}")

    # STEP 6 : Get the latent representation of each borrower
    # -----------------------------------------------------------------
    borrower_latent = encoder.predict(borrower_scaled)
    print("\nborrower_latent\n", borrower_latent[:5, :])
    print("\nborrower_latent shape\n", borrower_latent.shape)

    # STEP 6 : Cluster borrowers
    # -----------------------------------------------------------------
    cluster_labels = cluster_borrowers(borrower_latent, n_clusters=4)

    # STEP 7 : Add the cluster label to borrower data
    # -----------------------------------------------------------------
    borrowers_with_clusters = add_clusters_to_borrowers(borrower_df, cluster_labels)
    print("\nborrowers_with_clusters\n", borrowers_with_clusters.head(5))
    borrowers_with_clusters.to_excel(
        "borrowers_with_clusters.xlsx",
        index=False,
        header=True,
        engine="openpyxl"
    )


# ============================================
# Step 1: Generate random borrower data
# ============================================


def generate_borrower_data(n_samples: int):
    random_state = 42
    rng = np.random.default_rng(random_state)
    customer_ids = np.arange(1, n_samples + 1)

    # dataframe of dictionary
    borrower_df = pd.DataFrame({

        # Customer
        "customer_id": rng.permutation(customer_ids),

        # Financial
        "income": rng.normal(75000, 5000, n_samples).round(decimals=2),  # mean 75k
        "income_volatility": rng.uniform(0.05, 0.4, n_samples).round(decimals=4),  # relative variability
        "loan_amount": rng.normal(20000, 800, n_samples).round(decimals=2),
        "credit_limit": rng.normal(50000, 500, n_samples).round(2),

        # Behavioral
        "transaction_frequency": rng.integers(30, 70, n_samples),  # per month
        "missed_payment_count": rng.integers(0, 10, n_samples),  # in past 1 year

        # Demographic
        "age": rng.integers(21, 70, n_samples),
        "employment_type": rng.choice(["salaried", "self-employed", "contract"], n_samples),
        "region": rng.choice(["north", "south", "east", "west"], n_samples),

        # Account History
        "average_balance": rng.normal(10000, 500, n_samples).round(decimals=2),
        "account_tenure": rng.uniform(1, 15, n_samples).round(decimals=2),
    })

    # cap loan_amount at credit_limit
    borrower_df["loan_amount"] = np.minimum(borrower_df["loan_amount"], borrower_df["credit_limit"]).round(2)
    # compute debt_to_income = loan_amount / income
    borrower_df["debt_to_income"] = (borrower_df["loan_amount"] / borrower_df["income"]).round(4)
    # compute credit_utilization = loan_amount / credit_limit
    borrower_df["credit_utilization"] = (borrower_df["loan_amount"] / borrower_df["credit_limit"]).round(4)
    # generate loan_repayment_outstanding such that it never exceeds loan_amount
    # Example: draw a random fraction [0,1] of loan_amount
    fractions = rng.uniform(0.0, 1.0, n_samples)
    borrower_df["loan_amount_outstanding"] = (fractions * borrower_df["loan_amount"]).round(2)

    print("\nborrower_df\n", borrower_df)
    return borrower_df


# ============================================
# Encode categorical features and scale all features equally
# ============================================
def pre_process(borrower_df: pd.DataFrame):
    # 1. one-hot encoding to encode categorical columns to numerical
    borrower_df_unencoded = borrower_df.copy()
    borrower_df_encoded = one_hot_encoder(borrower_df_unencoded)

    # 2. scale all features for equal importance
    scaler = StandardScaler()
    borrower_scaled = scaler.fit_transform(borrower_df_encoded)

    return borrower_df_encoded, borrower_scaled


# ============================================
# One hot encoder
# ============================================


def one_hot_encoder(borrower_df_unencoded: pd.DataFrame):
    borrower_df_unencoded = borrower_df_unencoded.copy()
    categorical_cols = borrower_df_unencoded.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = borrower_df_unencoded.select_dtypes(include=['number']).columns.tolist()

    encoder = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ],
        remainder='passthrough'
    )

    borrower_df_encoded = encoder.fit_transform(borrower_df_unencoded)
    encoded_feature_names = (
        encoder.named_transformers_["cat"].get_feature_names_out(categorical_cols)
    )
    encoded_feature_names = list(encoded_feature_names) + numerical_cols
    borrower_df_encoded = pd.DataFrame(borrower_df_encoded, columns=encoded_feature_names)

    print("\ncategorical_cols\n", categorical_cols)
    print("\nnumerical_cols\n", numerical_cols)
    print("\nencoded_feature_names\n", encoded_feature_names)

    return borrower_df_encoded


# ============================================
# Train autoencoder to reduce the error in reconstructed input
# ============================================


def train_autoencoder(x_scaled, autoencoder, epochs, batch_size):
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    x_noisy = x_scaled + np.random.normal(0, 0.1, x_scaled.shape)

    history = autoencoder.fit(
        # input - with added noise for better reconstruction
        x_noisy,
        # x_scaled,

        # target - reconstruct from input
        x_scaled,

        # No of passes to train the network
        epochs=epochs,

        # Process records in batch size at a time
        batch_size=batch_size,

        # Feed random data order
        shuffle=True,

        # 10% of the data for validation
        validation_split=0.1,

        # verbose=1 → Prints progress bar and loss per epoch
        # verbose=0 → Silent training (no output)
        # verbose=2 → One line per epoch (no progress bar)
        verbose=2,

        # lets the model train longer but stops automatically when loss plateaus
        callbacks=[early_stop]
    )

    return history


# ============================================
# Build the autoencoder / neural network
# ============================================


def build_autoencoder(input_dim: int, encoding_dim: int):
    h1_encoding = encoding_dim * 3
    h2_encoding = encoding_dim * 2

    # Step1: Encoder :learns which features matter

    encoder_input = layers.Input(shape=(input_dim,), name="encoder_input")
    # first compression
    x = layers.Dense(h1_encoding, activation="relu", name="encoder_h1")(encoder_input)
    x = layers.BatchNormalization(name="bn_encoder_h1")(x)
    # second compression
    x = layers.Dense(h2_encoding, activation="relu", name="encoder_h2")(x)
    x = layers.BatchNormalization(name="bn_encoder_h2")(x)
    # latent layer
    latent = layers.Dense(encoding_dim, activation="relu", name="latent_layer")(x)
    latent = layers.BatchNormalization(name="bn_latent")(latent)

    # encoder
    encoder = models.Model(inputs=encoder_input, outputs=latent, name="encoder")

    # Step2: Decoder part — reconstructs original data

    # first decompression
    x = layers.Dense(h1_encoding, activation='relu', name="decoder_h1")(latent)
    x = layers.BatchNormalization(name="bn_decoder_h1")(x)
    # second decompression
    x = layers.Dense(h2_encoding, activation='relu', name="decoder_h2")(x)
    x = layers.BatchNormalization(name="bn_decoder_h2")(x)
    # reconstruct original layer
    decoder_output = layers.Dense(input_dim, activation='linear')(x)  # The final layer’s purpose is not to detect
    # patterns, but to output a reconstruction ideally an exact numeric match to the input features (after scaling).

    # decoder
    decoder = models.Model(inputs=latent, outputs=decoder_output, name="decoder")

    # Combine encoder + decoder
    autoencoder = models.Model(inputs=encoder_input, outputs=decoder_output, name="autoencoder")

    return autoencoder, encoder, decoder

# ============================================
# Allocate the cluster labels
# ============================================


def cluster_borrowers(x_latent, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(x_latent)
    return cluster_labels


# ============================================
# Add cluster labels to borrower
# ============================================

def add_clusters_to_borrowers(df: pd.DataFrame, cluster_labels: np.ndarray):
    df = df.copy()
    df["cluster"] = cluster_labels
    return df


if __name__ == "__main__":
    main()
