from keras import layers, models, ops
import numpy as np


# Step 1: Sampling layer (reparameterization trick)
class Sampling(layers.Layer):
    """Custom Keras Layer to sample z ~ N(mean, exp(log_var))"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = ops.random.normal(shape=ops.shape(z_mean))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


# Step 2: Build Encoder
def build_encoder(input_dim: int, latent_dim: int):
    inputs = layers.Input(shape=(input_dim,), name="encoder_input")
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="relu")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = models.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


# Step 3: Build Decoder
def build_decoder(latent_dim: int, output_dim: int):
    latent_inputs = layers.Input(shape=(latent_dim,), name="z_sampling")
    x = layers.Dense(32, activation="relu")(latent_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(output_dim, activation="sigmoid")(x)
    decoder = models.Model(latent_inputs, outputs, name="decoder")
    return decoder


# Step 4: Combine into a VAE model
def build_vae(input_dim: int, latent_dim: int):
    encoder = build_encoder(input_dim, latent_dim)
    decoder = build_decoder(latent_dim, input_dim)

    inputs = layers.Input(shape=(input_dim,), name="vae_input")
    z_mean, z_log_var, z = encoder(inputs)
    reconstructed = decoder(z)

    # ✅ Define a Keras layer that computes and adds the loss internally
    class VAELossLayer(layers.Layer):
        def call(self, inputs, reconstructed):
            z_mean, z_log_var = inputs
            # Use Keras ops to compute losses symbolically
            reconstruction_loss = ops.mean(ops.square(inputs_vae - reconstructed)) * input_dim
            kl_loss = -0.5 * ops.mean(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            self.add_loss(reconstruction_loss + kl_loss)
            return reconstructed

    # Important: bind symbolic tensor for reconstruction loss
    inputs_vae = inputs  # symbolic alias used inside custom layer

    outputs = VAELossLayer()( [z_mean, z_log_var], reconstructed )
    vae = models.Model(inputs, outputs, name="vae")

    # Compile — no explicit loss, as it’s embedded inside VAELossLayer
    vae.compile(optimizer="adam")
    return vae, encoder, decoder


# Step 5: Example usage
if __name__ == "__main__":
    input_dim = 18
    latent_dim = 4

    vae, encoder, decoder = build_vae(input_dim, latent_dim)
    vae.summary()

    X_train = np.random.rand(1000, input_dim)

    vae.fit(X_train, epochs=50, batch_size=32, validation_split=0.1)
