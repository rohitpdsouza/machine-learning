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
    borrower_df = pd.read_csv("data/input/borrower3.csv", header=0)
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
    
    Every epoch:
    1. Encoder pass
    For input x, encoder will output:
    z_mean -> mean of the latent distribution
    z_log_var -> log variance of the latent distribution
    
    2. Sampling pass
    Samples a probabilistic latent representation z of input using random noise from N(0,1)
    
    3. Decoder pass
    z is fed to the decoder network
    It reconstructs x_recon
    
    4. Compute loss
    Reconstruction loss (x & x_recon)
    kl_loss (how far is the latent representation z_mean & z_log i.e. encoder outputs from N(0,1)?)
    total loss = reconstruction loss + kl_loss (with a weight)
    
    5. Weight updates
    The total loss is used to update:
    encoder weights (which it used to compute z_mean, z_log_var, this will change the values in the next epoch)
    decoder weights (which improves reconstruction)
    """
    epochs = 100
    batch_size = 100

    feature_names = x_scaled_df.columns.tolist()
    history = train_vae(borrower_scaled, vae, epochs, batch_size, feature_names, encoder)
    print(f"VAE trained for {len(history.history['loss'])} epochs. Final loss: {history.history['loss'][-1]:.4f}")
    print("\nMetrics tracked:", history.history.keys(), "\n")

    # STEP 6 : Get the latent representation of each borrower
    # -----------------------------------------------------------------
    z_mean, z_log_var, z = encoder.predict(borrower_scaled)
    print("\nLatent borrower representation (first 5):\n", z[:5])
    print("\nLatent borrower shape: ", z.shape, "\n")

    # STEP 6 : Cluster borrowers
    # -----------------------------------------------------------------
    # cluster_labels = cluster_borrowers(z_mean, n_clusters=3)
    cluster_df = cluster_borrowers_gmm(z_mean, n_clusters=2)

    # print("\ncluster_labels\n", cluster_labels)
    print("\ncluster_labels\n", cluster_df.head(5))

    # STEP 7 : Add the cluster label to borrower data
    # -----------------------------------------------------------------
    # borrowers_with_clusters = add_clusters_to_borrowers(borrower_df, cluster_labels)

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
    # ignore some customer_id & region as they are not relevant
    # amplify credit utilization
    borrower_df_unencoded = borrower_df_unencoded.drop(columns=['customer_id', 'region'])
    borrower_df_unencoded['credit_util_power'] = borrower_df_unencoded['credit_utilization'] ** 10  # non-linear effect
    borrower_df_unencoded['credit_util_log'] = np.log1p(borrower_df_unencoded['credit_utilization'])  # if skewed

    borrower_df_encoded = one_hot_encoder(borrower_df_unencoded)

    # 2. scale features for equal importance
    scaler = StandardScaler()
    borrower_scaled = scaler.fit_transform(borrower_df_encoded)

    # Amplify credit utilization
    features = borrower_df_encoded.columns.tolist()
    idx = features.index('credit_utilization')
    borrower_scaled[:, idx] *= 2

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
    """
     Concept Check:
     Scalar : Single number / 0D tensor (5)
     Vector : An array of numbers / 1D tensor ([1,2,3])
     Matrix : A grid of numbers / 2D tensor ([1,2], [3,5])
     Tensor : scalars, vectors, and matrices to n dimensions
    """

    # Symbolic tensor to define the shape of model input
    encoder_input = layers.Input(shape=(input_dim,), name="encoder_input")

    # 1st dense layer
    """
    Dense layer object with 512 output units
    x is a KerasTensor representing the shape (batch_size, 512) 
    """
    x = layers.Dense(512, activation="relu", name="encoder_l1")(encoder_input)

    # 2nd dense layer
    """
    Dense layer object with 256 output units
    x is a KerasTensor representing the shape (batch_size, 256) 
    """
    x = layers.Dense(256, activation="relu", name="encoder_l2")(x)

    x = layers.Dense(128, activation="relu", name="encoder_l3")(x)
    x = layers.Dense(64, activation="relu", name="encoder_l4")(x)
    x = layers.Dense(32, activation="relu", name="encoder_l5")(x)

    # mean and variance layer
    """
    Encoder will learn the mean and log-variance in the latent space (batch size, latent_dim) for each borrower. In 
    variational encoder, borrower it not a point but a "gaussian cloud". Mean and log-variance will give the center 
    and spread of the cloud in gaussian space.
    
    Note that the below is just a linear transformation. At the beginning, z_mean and z_log_var can be both very 
    similar. They will diverge over each training step using the formula used in the VAE layer and eventually learn 
    the 'mean' and variance of the latent output.
    """
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
    x = layers.Dense(64, activation="relu", name="decoder_l2")(x)

    x = layers.Dense(128, activation="relu", name="decoder_l3")(x)
    x = layers.Dense(256, activation="relu", name="decoder_l4")(x)
    x = layers.Dense(512, activation="relu", name="decoder_l5")(x)

    # output layer
    decoder_output = layers.Dense(output_dim, activation="linear")(x)

    decoder = models.Model(latent_input, decoder_output, name="decoder")

    return decoder


def train_vae(x_scaled, vae, epochs, batch_size, feature_names, encoder):
    vae = vae

    # fist encoder layer is used for weights because it is the only layer that directly connects to the initial set
    # of raw borrower attributes
    callback_weights = FeatureWeightPrinter(
        encoder_model=encoder,
        feature_names=feature_names,
        dense_layer_name="encoder_l1",
        top_k=10
    )

    history = vae.fit(
        x_scaled,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[callback_weights],
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
        # epsilon is a tensor of random noise from standard distribution N(0,1)
        # each element of the tensor will be drawn from N(0,1)
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        # encoder generates the mean and log of the variance
        # tf.exp(0.5 * z_log_var) will compute the standard deviation
        # epsilon will give the random "jitter" from N(0,1)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAELossLayer(layers.Layer):
    """
    The constructor (__init__) runs when you create a VariationalAutoencoder(...) object.
    **kwargs: any extra arguments passed to the parent class.
    """

    """
    z_log_var is the logarithm of variance 
    σ is the standard deviation, which is derived from variance 
    Variance : σ² = exp(z_log_var)
    Standard deviation: σ = exp(0.5*z_log_var)
    """

    """
    VAE concept:
    
    At initialization, dense layer will randomly assign weights using linear transformation. They can be (are) similar
    
    z_mean = 0.1
    z_log_var = 0.1
    variance σ² = exp(z_log_var) = e(0.1) = 1.105
    SD σ = sqrt(1.105) = 1.05
    
    Sample a latent value using the mean, variance and random noise (epsilon ε)
    random noise is sampled from standard deviation N(0,1), assume ε = 0.5
    z = z_mean + σ * ε
      = 0.1 + 1.05 * 0.5
      = 0.625
    
    Training step1:
    input = 10
    reconstructed = 15
    error = (15-10)power(2) = 25
    
    To reduce the reconstruction loss, neural network will push z_mean 
    Say new z_mean = 0.8 (was initially 0.1)
    
    kl_loss wants to shrink towards N(0,1), pushes mean -> 0 and variance ->1
    say updated z_log_var = -0.2 (from 0.1)
    
    Now:
    z_mean = 0.8
    z_log_var = -0.2
    variance σ² = exp(z_log_var) = e(-0.2) = 0.818
    SD σ = sqrt(0.818) = 0.904
    Sampling:
    z = z_mean + σ * ε
    = 0.8 + 0.904 * 0.3
    = 1.07
    
    Training step2:
    input = 10
    reconstructed = 12
    error = 4
    
    To push reconstructed towards input (reduce reconstruction loss), VAE will shift z_mean
    z_mean = 1.3 (from 0.8)
    
    kl_loss will push variation towards N(0,1).
    updated z_log_var = -0.6
    Now:
    z_mean = 1.3
    z_log_var = -0.6
    variance σ² = exp(z_log_var) = e(-0.6) = 0.548
    SD σ = sqrt(0.548) = 0.740
    Sampling:
    z = z_mean + σ * ε
    = 1.3 + 0.740 * (-0.2)
    = 1.152
      
    """

    def __init__(self, beta=0.001, **kwargs):
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
        # KL divergence term measures how far each borrower’s encoded
        # distribution is from the standard normal distribution
        kl_loss = -0.5 * ops.mean(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))

        # Add combined loss
        self.add_loss(reconstruction_loss + self.beta * kl_loss)

        # Return the reconstructed tensor (for model output)
        return reconstructed


class FeatureWeightPrinter(tf.keras.callbacks.Callback):
    """
    A callback that prints the learned feature weights of a specific Dense layer
    inside the encoder after each epoch.

    Parameters
    ----------
    encoder_model : keras.Model
        The Keras encoder model that contains Dense layers.

    feature_names : list[str]
        List of feature names corresponding to input columns.

    dense_layer_name : str
        The name of the Dense layer whose weights should be printed.

    top_k : int
        Number of highest-magnitude feature weights to display each epoch.
    """

    def __init__(self, encoder_model, feature_names, dense_layer_name, top_k=10):
        super().__init__()
        self.encoder_model = encoder_model  # <--- THIS IS A MODEL OBJECT (TENSOR-BASED)
        self.feature_names = feature_names  # <--- PLAIN PYTHON LIST
        self.dense_layer_name = dense_layer_name  # <--- PLAIN STRING
        self.top_k = top_k  # <--- PLAIN INTEGER

    def on_epoch_end(self, epoch, logs=None):
        """Executed automatically after every epoch."""

        # ---- 1. Get the Dense layer inside the encoder ----
        dense_layer = self.encoder_model.get_layer(self.dense_layer_name)

        # ---- 2. Extract its weights ----
        W, b = dense_layer.get_weights()
        # W shape = (num_features, num_neurons)

        # ---- 3. Use the first neuron’s weight vector (vector of size num_features) ----
        weight_vector = W[:, 0]

        # ---- 4. Pair feature names with their weights ----
        feature_weight_pairs = list(zip(self.feature_names, weight_vector))

        # ---- 5. Sort by absolute value (importance) ----
        feature_weight_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        # ---- 6. Print clean summary ----
        print(f"\n=== Epoch {epoch + 1} — Top {self.top_k} Feature Weights ===")
        for name, weight in feature_weight_pairs[:self.top_k]:
            print(f"{name:30s} → {weight:+.4f}")


if __name__ == "__main__":
    main()
