"""
A variational autoencoder (VAE) is a more advanced neural network that learns not just a single compressed
version, but a probability range (mean and variance) for each borrower’s features — allowing it to understand
uncertainty and naturally group similar borrowers (e.g., low, medium, or high risk) in a smooth, continuous space.
"""

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import layers, models, ops
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
    # Read borrower dataset
    borrower_df = pd.read_csv("data/input/borrower2.csv", header=0)
    print("\nborrower_df\n", borrower_df.shape)

    # STEP 2: Encode categorical features and scale all features equally
    # -----------------------------------------------------------------
    borrower_df_encoded, borrower_scaled = pre_process(borrower_df)
    x_scaled_df = pd.DataFrame(borrower_scaled, columns=borrower_df_encoded.columns.tolist())
    print("\nborrower_df encoded\n", borrower_df_encoded.head(5))
    print("\nborrower_df scaled\n", x_scaled_df.head(5))

    print("\nborrower_df encoded shape:\n", borrower_df_encoded.shape)
    print("\nborrower_df scaled shape:\n", x_scaled_df.shape)

    # STEP 3: Build variational autoencoder (VAE) to compress and de-compress the input and output
    # -----------------------------------------------------------------
    input_dim = borrower_scaled.shape[1]
    encoding_dim = 5  # latent space size
    print("\nNo of input features: ", input_dim, "\n")
    print("\nNo of latent features: ", encoding_dim, "\n")
    vae, encoder, decoder = build_variational_autoencoder(input_dim, encoding_dim)

    # STEP 4 : Compile the VAE
    # -----------------------------------------------------------------
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    # STEP 5 : Train autoencoder
    # -----------------------------------------------------------------
    """
    The model tries to minimize the difference between input and reconstructed output and 
    automatically learns the feature weights in the process
    """
    epochs = 200
    batch_size = 100

    history = train_vae(borrower_scaled, vae, epochs, batch_size)
    print(f"VAE trained for {len(history.history['loss'])} epochs. Final loss: {history.history['loss'][-1]:.4f}")
    print("\nMetrics tracked:", history.history.keys(), "\n")

    # STEP 6 : Get the latent representation of each borrower
    # -----------------------------------------------------------------
    z_mean, z_log_var, z = encoder.predict(borrower_scaled)
    print("\nLatent borrower representation (first 5):\n", z[:5])
    print("\nLatent borrower shape: ", z.shape, "\n")

    # STEP 6 : Cluster borrowers
    # -----------------------------------------------------------------
    #cluster_labels = cluster_borrowers(z_mean, n_clusters=3)
    cluster_df = cluster_borrowers_gmm(z_mean, n_clusters=4)

    #print("\ncluster_labels\n", cluster_labels)
    print("\ncluster_labels\n", cluster_df.head(5))

    # STEP 7 : Add the cluster label to borrower data
    # -----------------------------------------------------------------
    #borrowers_with_clusters = add_clusters_to_borrowers(borrower_df, cluster_labels)

    borrowers_with_clusters = pd.concat([borrower_df.reset_index(drop=True), cluster_df], axis=1)

    print("\nborrowers_with_clusters\n", borrowers_with_clusters.head(5))
    borrowers_with_clusters.to_excel(
        "data/output/borrowers_with_clusters_vae_2.xlsx",
        index=False,
        header=True,
        engine="openpyxl"
    )


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


def build_variational_autoencoder(input_dim: int, latent_dim: int):
    """
    A variational encoder will give a probability distribution (mean and variance) of the latent borrower attributes.

    It means that the specific latent borrower attribute can take any value in the (mean, variance) space -
    this is what is called as 'probabilistic or stochastic' determination.

    Borrowers with similar distribution (mean, variance) in their latent attributes will lie close together and
    are then grouped into clusters using clustering algorithm
    """
    encoder = build_encoder(input_dim, latent_dim)
    decoder = build_decoder(latent_dim, input_dim)

    inputs = layers.Input(shape=(input_dim,), name="vae_input")
    z_mean, z_log_var, z = encoder(inputs)
    reconstructed = decoder(z)

    outputs = VAELossLayer()([z_mean, z_log_var, inputs, reconstructed])

    vae = models.Model(inputs, outputs, name="vae")

    return vae, encoder, decoder


def build_encoder(input_dim, latent_dim):
    # Symbolic tensor to define the shape of model input
    encoder_input = layers.Input(shape=(input_dim,), name="encoder_input")

    # 1st dense layer
    """
    Dense layer object with 64 output units
    x is a KerasTensor representing the shape (batch_size, 64) 
    """
    x = layers.Dense(256, activation="relu", name="encoder_l1")(encoder_input)

    x = layers.Dense(128, activation="relu", name="encoder_l4")(x)
    x = layers.Dense(64, activation="relu", name="encoder_l3")(x)

    # 2nd dense layer
    """
    Dense layer object with 32 output units
    x is a KerasTensor representing the shape (batch_size, 32) 
    """
    x = layers.Dense(32, activation="relu", name="encoder_l2")(x)

    # mean and variance layer
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    # Encoder model
    encoder = models.Model(inputs=encoder_input, outputs=[z_mean, z_log_var, z], name="encoder")

    return encoder


def build_decoder(latent_dim, output_dim):
    # Symbolic tensor to define the shape of the model input
    latent_input = layers.Input(shape=(latent_dim,), name="latent_sampling")

    # 1st dense layer
    x = layers.Dense(32, activation="relu", name="decoder_l1")(latent_input)

    # 2nd dense layer
    x = layers.Dense(64, activation="relu", name="decoder_l3")(x)

    x = layers.Dense(128, activation="relu", name="decoder_l2")(x)

    x = layers.Dense(256, activation="relu", name="decoder_l4")(x)

    # output layer
    decoder_output = layers.Dense(output_dim, activation="linear")(x)

    decoder = models.Model(latent_input, decoder_output, name="decoder")

    return decoder


def train_vae(x_scaled, vae, epochs, batch_size):
    vae = vae

    history = vae.fit(
        x_scaled,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1
    )

    return history


def cluster_borrowers_gmm(
    z_latent: np.ndarray,
    n_clusters: int,
    covariance_type: str = "full",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Cluster borrowers using Gaussian Mixture Model (GMM)
    on latent representations from a Variational Autoencoder.

    Args:
        z_latent (np.ndarray): Latent borrower features (z_mean from VAE encoder).
        n_clusters (int): Expected number of clusters (e.g., 3 = Low, Medium, High Risk).
        covariance_type (str): Type of covariance ("full", "tied", "diag", "spherical").
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Borrowers with GMM cluster labels and probabilities.
    """

    # Step 1: Initialize GMM
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        random_state=random_state
    )

    # Step 2: Fit and predict cluster memberships
    cluster_labels = gmm.fit_predict(z_latent)

    # Step 3: Soft cluster probabilities (posterior probabilities)
    cluster_probs = gmm.predict_proba(z_latent)

    # Step 4: Compute silhouette score (optional, for evaluation)
    try:
        silhouette = silhouette_score(z_latent, cluster_labels)
    except Exception:
        silhouette = np.nan

    # Step 5: Build results DataFrame
    cluster_df = pd.DataFrame({
        "cluster_id": cluster_labels,
        "cluster_confidence": cluster_probs.max(axis=1)  # Highest membership probability per borrower
    })

    print(f"✅ GMM found {len(np.unique(cluster_labels))} clusters.")
    print(f"📈 Average cluster confidence: {cluster_df['cluster_confidence'].mean():.3f}")
    print(f"🔹 Silhouette score (cluster separation): {silhouette:.3f}")

    return cluster_df


def cluster_borrowers(x_latent, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(x_latent)
    return cluster_labels


def add_clusters_to_borrowers(df: pd.DataFrame, cluster_labels: np.ndarray):
    df = df.copy()
    df["cluster"] = cluster_labels
    return df


class Sampling(layers.Layer):
    """Custom Keras Layer to sample z ~ N(mean, exp(log_var))"""

    def __init__(self, **kwargs):
        # Calls the constructor of the parent class (tf.keras.Model) so internal Keras setup happens correctly.
        # Always do this when subclassing Keras models.
        super().__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAELossLayer(layers.Layer):
    """
    The constructor (__init__) runs when you create a VariationalAutoencoder(...) object.
    **kwargs: any extra arguments passed to the parent class.
    """
    def __init__(self, beta=0.01, **kwargs):
        # Calls the constructor of the parent class (tf.keras.Model) so internal Keras setup happens correctly.
        # Always do this when subclassing Keras models.
        super().__init__(**kwargs)
        self.beta = tf.Variable(beta, trainable=False, dtype=tf.float32)

    def call(self, inputs):
        # Keras tensors
        z_mean, z_log_var, original_inputs, reconstructed = inputs

        input_dim = ops.shape(original_inputs)[-1]

        # Use Keras ops to compute losses symbolically
        reconstruction_loss = ops.mean(ops.square(original_inputs - reconstructed)) * ops.cast(input_dim, "float32")

        # Compute KL divergence loss (symbolic)
        kl_loss = -0.5 * ops.mean(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))

        # Add combined loss
        self.add_loss(reconstruction_loss + self.beta * kl_loss)

        # Return the reconstructed tensor (for model output)
        return reconstructed


if __name__ == "__main__":
    main()
