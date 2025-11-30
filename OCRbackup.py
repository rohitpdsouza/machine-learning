# crnn_ocr.py
# Python 3.11+, TensorFlow 2.x
from __future__ import annotations
from typing import List, Tuple, Dict

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------------------------------------------------------
# Config / Character set
# -----------------------------------------------------------------------------
IMG_H = 32            # fixed image height
IMG_W = 256           # fixed image width (pad/truncate images to this)
BATCH_SIZE = 32
CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-.,:;!?()/ "  # customize
BLANK_LABEL = ""      # CTC blank handled internally
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank token

# Map char -> int and inverse
char_to_num = {c: i for i, c in enumerate(CHARS)}
num_to_char = {i: c for c, i in char_to_num.items()}


# -----------------------------------------------------------------------------
# Utilities: label encoding / padding
# -----------------------------------------------------------------------------
def text_to_labels(text: str) -> List[int]:
    """Convert text string to list of label indices."""
    return [char_to_num[c] for c in text if c in char_to_num]


def labels_to_text(labels: List[int]) -> str:
    """Convert sequence of label ints to string (no collapsing/blanks)."""
    return "".join(num_to_char[i] for i in labels if i in num_to_char)


# -----------------------------------------------------------------------------
# Model builder: CRNN
# -----------------------------------------------------------------------------
def build_crnn(input_shape: Tuple[int, int, int] = (IMG_H, IMG_W, 1),
               num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Build the base CRNN model that outputs per-timestep class probabilities.
    This model is used both for training (wrapped with CTC) and inference.
    """
    inputs = layers.Input(shape=input_shape, name="image_input")  # (H,W,1)

    # --- CNN feature extractor
    x = inputs
    # conv block 1
    x = layers.Conv2D(64, kernel_size=3, padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)  # H/2, W/2

    # conv block 2
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)  # H/4, W/4

    # conv block 3 (preserve more width resolution)
    x = layers.Conv2D(256, kernel_size=3, padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Note: pool only in height to preserve more time-steps for sequence
    x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding="same")(x)  # H/8, W/4

    # conv block 4
    x = layers.Conv2D(512, kernel_size=3, padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), padding="same")(x)  # H/8, W/8

    # Now shape ~ (batch, H', W', C)
    shape = tf.keras.backend.int_shape(x)
    # Time dimension will be width axis
    # Collapse height and channels into features per time-step
    # Permute -> (batch, width, height*channels)
    x = layers.Permute((2, 1, 3))(x)  # (batch, W', H', C)
    t_dim = tf.keras.backend.int_shape(x)[1]
    x = layers.Reshape((t_dim, -1))(x)  # (batch, time_steps, features)

    # --- RNN sequence modeling
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2))(x)

    # --- Output layer (per-timestep classification)
    x = layers.Dense(512, activation="relu")(x)
    y_pred = layers.Dense(num_classes, activation="softmax", name="y_pred")(x)

    model = tf.keras.Model(inputs=inputs, outputs=y_pred, name="crnn_base")
    return model


# -----------------------------------------------------------------------------
# Training model with CTC loss wrapper
# -----------------------------------------------------------------------------
def build_training_model(crnn_model: tf.keras.Model) -> tf.keras.Model:
    """
    Build a model for training that takes additional inputs:
      - labels: sparse/padded labels
      - input_length: number of time steps produced by network per sample
      - label_length: length of each label sequence
    and outputs the CTC loss (so we can compile and fit the Keras model).
    """
    image_input = crnn_model.input  # (H,W,1)
    y_pred = crnn_model.output       # (batch, time_steps, num_classes)

    # Additional training-only inputs
    labels = layers.Input(name="labels", shape=(None,), dtype="int32")          # padded label sequences
    input_length = layers.Input(name="input_length", shape=(1,), dtype="int32")
    label_length = layers.Input(name="label_length", shape=(1,), dtype="int32")

    # CTC loss uses y_pred in time-major? K.ctc_batch_cost expects (y_true, y_pred, input_length, label_length)
    # y_pred must be (batch, time_steps, num_classes)
    def ctc_lambda(args):
        y_pred, labels, input_length, label_length = args
        # tf.keras.backend.ctc_batch_cost returns shape (batch, 1)
        return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

    loss_out = layers.Lambda(ctc_lambda, output_shape=(1,), name="ctc_loss")(
        [y_pred, labels, input_length, label_length]
    )

    training_model = tf.keras.Model(
        inputs=[image_input, labels, input_length, label_length],
        outputs=loss_out,
        name="ocr_training_model"
    )
    # Compile with a dummy lambda loss because the model already outputs the loss value
    training_model.compile(optimizer=keras.optimizers.Adam(1e-4),
                           loss={"ctc_loss": lambda y_true, y_pred: y_pred})
    return training_model


# -----------------------------------------------------------------------------
# Decoding helpers for inference
# -----------------------------------------------------------------------------
def decode_predictions(y_pred: np.ndarray,
                       input_lengths: np.ndarray,
                       beam_width: int = 1) -> List[str]:
    """
    Decode model predictions to strings.
    - y_pred: (batch, time_steps, num_classes)
    - input_lengths: array of actual time_steps (for each sample) the network produced
    - beam_width: 1 -> greedy; >1 -> beam search (tf.keras's ctc_decode uses beam_width)
    """
    # Convert to logits for ctc_decode (it expects logits or probabilities; we pass probs)
    # tf.keras.backend.ctc_decode returns (decoded, log_prob)
    decoded, log_prob = tf.keras.backend.ctc_decode(y_pred, input_length=input_lengths, greedy=(beam_width==1), beam_width=beam_width, top_paths=1)
    decoded_indices = decoded[0].numpy()  # shape: (batch, seq_len)
    results = []
    for seq in decoded_indices:
        # seq contains ints referring to character indices; skip blanks and out-of-range
        text = "".join(num_to_char[i] for i in seq if i in num_to_char)
        results.append(text)
    return results


# -----------------------------------------------------------------------------
# Example data pipeline utilities (minimal)
# -----------------------------------------------------------------------------
def preprocess_image(img_path: str, img_h: int = IMG_H, img_w: int = IMG_W) -> np.ndarray:
    """
    Load grayscale image, resize (keep aspect ratio if desired), pad/truncate to (img_h, img_w)
    Returns float32 array in shape (img_h, img_w, 1), range [0,1]
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)  # 0..1
    # Resize maintaining aspect ratio, then pad to target width
    # For simplicity, resize to (img_h, proportional width), then pad/truncate to img_w
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    new_w = tf.cast(tf.math.ceil(img_w * (tf.cast(w, tf.float32) / tf.cast(h, tf.float32)) * (tf.cast(img_h, tf.float32) / tf.cast(img_h, tf.float32))), tf.int32)
    # Simpler: just resize to (img_h, img_w) - distorts some fonts but is easiest
    img = tf.image.resize(img, [img_h, img_w])
    return img.numpy().astype(np.float32)


def make_batch_from_paths(paths: List[str], texts: List[str]) -> Dict[str, np.ndarray]:
    """
    Create a single training batch (numpy arrays) from lists of image paths and label texts.
    Returns dict compatible with training_model inputs and a dummy y (zeros).
    """
    batch_images = np.stack([preprocess_image(p) for p in paths], axis=0)  # (B, H, W, 1)
    # encode labels
    label_seqs = [text_to_labels(t) for t in texts]
    label_lengths = np.array([len(s) for s in label_seqs], dtype=np.int32)
    max_lab = max(label_lengths) if len(label_lengths) > 0 else 0
    padded_labels = np.zeros((len(label_seqs), max_lab), dtype=np.int32)
    for i, s in enumerate(label_seqs):
        padded_labels[i, :len(s)] = s

    # input_length: number of time-steps produced by the network for each image
    # We must compute based on the CRNN's downsampling: roughly img_w / downsample_factor
    # For our CNN, approximate time_steps = IMG_W // 8 (because of strides/pools)
    time_steps = IMG_W // 8
    input_lengths = np.ones((len(paths), 1), dtype=np.int32) * time_steps
    label_lengths = label_lengths.reshape(-1, 1)

    # dummy zero y (training_model expects y_true; our loss lambda ignores it)
    dummy_y = np.zeros((len(paths), 1), dtype=np.float32)

    return {
        "image_input": batch_images,
        "labels": padded_labels,
        "input_length": input_lengths,
        "label_length": label_lengths
    }, dummy_y


# -----------------------------------------------------------------------------
# Putting it all together: build, train (example), inference example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Build base model and training wrapper
    crnn = build_crnn((IMG_H, IMG_W, 1), num_classes=NUM_CLASSES)
    crnn.summary()

    training_model = build_training_model(crnn)
    training_model.summary()

    # ---------- Example: single-batch training step (replace with real dataset) ----------
    # Suppose you have lists of image paths and corresponding texts:
    example_paths = ["/mnt/data/5802708c-e5cd-446e-951a-61a19cd123db.png"] * 2  # demo; replace with your dataset paths
    example_texts = ["Hello", "World1"]  # ground truth strings for those images

    X_batch, y_dummy = make_batch_from_paths(example_paths, example_texts)

    # Fit a single epoch (for demo). For realistic training use tf.data.Dataset and many epochs.
    training_model.fit(
        X_batch,
        y_dummy,
        batch_size=len(example_paths),
        epochs=1
    )

    # ---------- Inference ----------
    # Use the base `crnn` model for prediction, then decode with CTC
    images = np.stack([preprocess_image(p) for p in example_paths], axis=0)
    y_probs = crnn.predict(images)  # shape (B, time_steps, num_classes)

    input_lengths = np.ones((len(images),), dtype=np.int32) * (IMG_W // 8)
    decoded = decode_predictions(y_probs, input_lengths, beam_width=1)
    print("Decoded:", decoded)
