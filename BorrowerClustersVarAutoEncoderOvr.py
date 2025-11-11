import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import layers, models
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
    x_scaled_df = pd.DataFrame(borrower_scaled, columns=borrower_df_encoded.columns.tolist())
    print("\nborrower_df scaled\n", x_scaled_df)

    # STEP 3: Build variational autoencoder (VAE) to compress and de-compress the input and output
    # -----------------------------------------------------------------
    input_dim = borrower_scaled.shape[1]
    print("\nShape of scaled borrower_df:", borrower_scaled.shape, "\n")
    encoding_dim = 6  # latent space size
    vae, encoder, decoder = build_variational_autoencoder(input_dim, encoding_dim)

    # STEP 4 : Compile the VAE
    # -----------------------------------------------------------------
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    # STEP 5 : Train autoencoder
    # -----------------------------------------------------------------
    # The model tries to minimize the difference between input and reconstructed output and automatically learns the feature weights in the process
    epochs = 200
    batch_size = 64
    train_vae(borrower_scaled, vae, epochs, batch_size)

    # STEP 6 : Get the latent representation of each borrower
    # -----------------------------------------------------------------
    z_mean, z_log_var, z = encoder.predict(borrower_scaled)
    print("\nLatent borrower representation (first 5):\n", z[:5])
    print("\nLatent borrower shape: ", z.shape, "\n")

    # STEP 6 : Cluster borrowers
    # -----------------------------------------------------------------
    cluster_labels = cluster_borrowers(z, n_clusters=4)

    # STEP 7 : Add the cluster label to borrower data
    # -----------------------------------------------------------------
    borrowers_with_clusters = add_clusters_to_borrowers(borrower_df, cluster_labels)
    print("\nborrowers_with_clusters\n", borrowers_with_clusters)
    borrowers_with_clusters.to_excel(
        "borrowers_with_clusters_vae.xlsx",
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
    print("\nborrower_df encoded\n", borrower_df_encoded)

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
    print("\ncategorical_cols\n", categorical_cols)
    print("\nnumerical_cols\n", numerical_cols)

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
    print("\nencoded_feature_names\n", encoded_feature_names)
    borrower_df_encoded = pd.DataFrame(borrower_df_encoded, columns=encoded_feature_names)

    return borrower_df_encoded


# ============================================
# Train autoencoder to reduce the error in reconstructed input
# ============================================

def train_vae(x_scaled, vae, epochs, batch_size):
    vae = vae
    history = vae.fit(
        x_scaled,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1)

    print(f"VAE trained for {len(history.history['loss'])} epochs. Final loss: {history.history['loss'][-1]:.4f}")


# ============================================
# Build the autoencoder / neural network
# ============================================

def build_variational_autoencoder(input_dim: int, encoding_dim: int):
    # ------Step1: Encoder :learns which features matter-------
    encoder_inputs = layers.Input(shape=(input_dim,), name="encoder_inputs")

    # first dense layer - 64 neurons. Learn 64 combinations from the existing borrower attributes
    x = layers.Dense(64, activation="relu", name="encoder_64")(encoder_inputs)

    # second smaller layer - 32 neurons. Compress to 32 combinations.
    x = layers.Dense(32, activation="relu", name="encoder_32")(x)

    """
    A variational encoder will give a probability distribution (mean and variance) of the latent borrower attributes. 
    
    It means that the specific latent borrower attribute can take any value in the (mean, variance) space - 
    this is what is called as 'probabilistic or stochastic' determination. 
    
    Borrowers with similar distribution (mean, variance) in their latent attributes will lie close together and 
    are then grouped into clusters using clustering algorithm
    """

    # mean and variance layer
    z_mean = layers.Dense(encoding_dim, name="z_mean")(x)
    z_log_var = layers.Dense(encoding_dim, name="z_log_var")(x)
    # Reparameterization
    z =Sampling()([z_mean, z_log_var])

    # encoder
    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # --------Step2: Decoder part — reconstructs original data-------

    # first decompression
    latent_inputs = layers.Input(shape=(encoding_dim,), name="latent_sampling")
    x = layers.Dense(32, activation="relu", name="decoder_32")(latent_inputs)
    x = layers.Dense(64, activation="relu", name="decoder_64")(x)
    # reconstruct original layer
    decoder_outputs = layers.Dense(input_dim, activation="linear")(x)

    #decoder
    decoder = models.Model(latent_inputs, decoder_outputs, name="decoder")

    # Full VAE
    vae = VariationalAutoencoder(encoder, decoder, input_dim)

    return vae, encoder, decoder


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

# ============================================
# Sampling
# ============================================

class Sampling(layers.Layer):
    """
    Reparameterization trick: z = mean + exp(0.5 * log_var) * eps
    The Sampling layer performs the "reparameterization trick":
    z = mean + exp(0.5 * log_var) * epsilon
    - z_mean: average location of the latent distribution
    - z_log_var: log of the variance (used for numerical stability)
    - epsilon: random noise ~ N(0,1)
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# -----------------------------------
# Variational Autoencoder Model
# -----------------------------------
"""
Defines a new class, it inherits from tf.keras.Model so it behaves like any Keras model (can be compile()-ed, fit()-ed, 
predict()-ed).
"""

class VariationalAutoencoder(tf.keras.Model):
    """
    The constructor (__init__) runs when you create a VariationalAutoencoder(...) object.
    **kwargs: any extra arguments passed to the parent class.
    """
    def __init__(self, encoder, decoder, input_dim, **kwargs):

        # Calls the constructor of the parent class (tf.keras.Model) so internal Keras setup happens correctly.
        # Always do this when subclassing Keras models.
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.input_dim = input_dim

        # Creates three metric objects (Mean) that will keep running averages during training
        """ average of the combined loss (reconstruction + KL) """
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        """ average of reconstruction MSE """
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        """ average of KL divergence """
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    # This defines the metrics property that Keras reads automatically.
    # By listing these metrics here, Keras knows to reset them at the start of each epoch and include them in logs.
    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    # Overrides how a single training step (one batch) is computed
    # Keras calls this automatically inside model.fit() for each batch.

    def train_step(self, data):
        """Custom training step for VAE."""
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Reconstruction loss: how well we rebuild input
            reconstruction_loss = tf.reduce_mean(tf.square(data - reconstruction)) * self.input_dim

            # KL divergence loss: how close latent space is to N(0,1)
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

            total_loss = reconstruction_loss + kl_loss

        # This is the step that updates the weights to reduce loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Track losses
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

if __name__ == "__main__":
    main()


