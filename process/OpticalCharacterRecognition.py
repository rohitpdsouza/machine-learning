"""
Write a basic CNN (convolutional neural network) and RNN (recurrent neural network) to understand how character
recognition works.

TODO: Due to the CTC collapse rule, repeated characters currently collapse in this neural network. e.g. "hello" is
decoded as "helo". Also, the model is able to decode "fig" but not "program".
Next, train the model on an enriched list of words, so the model learns more word patterns
"""
import os
import tensorflow as tf
from keras.src.utils import load_img
from tensorflow.keras import layers, models
import numpy as np
import cv2

# Set the working directory to the ML folder to ensure that all file paths are relative to this directory.
os.chdir("C:/Users/prohi/PycharmProjects/POC/ML")


def pre_process(img, img_width, img_height):
    # Resizing makes every character sequence fit the same feature extraction scale
    """
    Most OCR architectures assume a long-width but short-height shape:
    128 × 32 → CRNN / Tesseract training
    100 × 32 → IAM handwriting OCR
    256 × 32 → scene text models
    512 × 32 / 512 × 64 → printed OCR
    The height is kept small (32 or 64) because the vertical variation in handwriting is low.
    """
    img = img.resize((img_width, img_height))

    # normalize 0–1 : TensorFlow/Keras models work with NumPy arrays (tensors), not PIL images
    """
    Pixel values originally are 0–255 integers
    Normalizing converts them to 0–1 floats.
    CNNs always expect normalized input. Normalization is a standard preprocessing step.
    """
    img_arr = np.array(img).astype("float32") / 255.0
    # CNN layers expect an image tensor shaped as: (batch_size, height, width, channel).
    # For grayscale → channels = 1
    # Keras Conv2D requires a channel dimension (e.g., 1 for grayscale, 3 for RGB).
    # For a single image, batch size = 1
    img_arr = np.expand_dims(img_arr, axis=-1)  # add 1 channel dimension at the end
    img_arr = np.expand_dims(img_arr, axis=0)  # add 1 batch dimension at the start

    print("Input image tensor shape:", img_arr.shape)
    """
    Note that PIL resize uses (width, height) -> hence (128, 32)
    TensorFlow uses (batch size, height, width, channel) -> hence (1, 32, 128, 1)
    """

    return img_arr


def build_cnn_feature_extractor(img_arr):
    """
    :param img_arr:
    :return:

    32 in Conv2D means layer uses 32 filters (also called kernels) Each filter learns a different pattern,
    such as an edge, a diagonal stroke, a curve, intersection of lines, shape pattern etc. The output is 32 feature maps

    (3x3) is 3x3 sliding window that moves across the image, standard for CNN architecture
    It slides over every pixel in the input and looks at its 3x3 neighborhood to detect visual features

    padding=same means do not shrink the image, input and output are both 32 x 128

    relu is the non-linear activation to make the neural network learn non-linear patterns

    32 filters will produce an output of (32, 128, 32). Each filter will detect a different visual pattern.
    E.g.
    filter1 -> detects vertical lines
    filter2 -> detects horizontal edges
    filters -> detects curves
    etc...

    MaxPooling2D(2,2) shrinks the image by half while preserving only the strongest visual features
    Pooling is necessary in OCR to :
    1) Reduce computation : lesser pixels -> faster training
    2) Keep the strongest features: edges/strokes stay, noise goes away
    3) Make the features more scale-invariant: if the same letter appears slightly larger or smaller, pooling detects it
    4) Prepare data for the LSTM step: CNN output must be converted to a sequence
    """
    img_height = img_arr.shape[1]
    img_width = img_arr.shape[2]

    # Height, Width, Channel
    inputs = layers.Input(shape=(img_height, img_width, 1))

    # Convolution 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Convolution 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Convolution 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Convolution 4
    outputs = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_feature_extractor")
    return model


def build_rnn_sequence_model(cnn_model):
    """
    Reshape the CNN output tensor (batch,height,width,channels) into (batch, timesteps=width, features=height*channels)
    Two bidirectional LSTM layers learn left-to-right context and right-to-left context
    Output of RNN model is (batch, width, channels*2). e.g. 256 forward and 256 backward

    The number of neurons in the LSTM layers defines the memory capacity for reading sequences. 256 is the standard.
    CNN channels gives the number of visual patterns
    """
    cnn_output = cnn_model.output
    print(f"CNN output feature maps tensor shape {cnn_output.shape}\n")
    batch, height, width, channels = cnn_model.output.shape
    rnn_seq_input = layers.Reshape(target_shape=(width, height * channels))(cnn_output)
    print(f"Convert CNN feature maps into sequences for LSTM with tensor shape {rnn_seq_input.shape}\n")

    # First Bi-LSTM Layer
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=0.25)
    )(rnn_seq_input)

    # Second Bi-LSTM layer
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=0.25)
    )(x)

    # Rnn takes CNN feature map shape as input
    rnn_model = models.Model(inputs=cnn_model.output, outputs=x, name='rnn_sequencing_stage')

    return rnn_model


class CTCLossLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # important: call parent constructor

    """Adds CTC loss to the model."""

    def call(self, inputs):
        y_true, y_pred, input_len, label_len = inputs

        ctc_loss = tf.keras.backend.ctc_batch_cost(
            y_true, y_pred, input_len, label_len
        )

        self.add_loss(ctc_loss)
        return y_pred  # Pass predictions forward


def build_crnn_model(cnn_model, rnn_model, max_label_length, num_classes):
    img_input = layers.Input(
        shape=cnn_model.input_shape[1:],
        name="input_layer"
    )

    # Connect new input → CNN → (CNN output)
    cnn_out = cnn_model(img_input)

    # Connect CNN output → RNN → (RNN output)
    rnn_output = rnn_model(cnn_out)

    # Add Dense + Softmax here
    softmax_output = layers.Dense(
        num_classes,
        activation="softmax",
        name="softmax_output"
    )(rnn_output)

    # true label input
    y_true = layers.Input(
        name="y_true",
        shape=(max_label_length,),
        dtype="int32"
    )

    # required lengths
    input_len = layers.Input(name="input_length", shape=(1,), dtype="int32")
    label_len = layers.Input(name="label_length", shape=(1,), dtype="int32")

    loss_output = CTCLossLayer()([y_true, softmax_output, input_len, label_len])

    crnn_model = models.Model(
        inputs=[img_input, y_true, input_len, label_len],
        outputs=loss_output,
        name="crnn_ctc_model"
    )

    return crnn_model


def build_inference_model(crnn_model):
    # Get the image input only (the first input of crnn_model)
    image_input = crnn_model.inputs[0]

    # Get the softmax output layer
    softmax_layer = crnn_model.get_layer("softmax_output")

    inference_model = models.Model(
        inputs=image_input,
        outputs=softmax_layer.output,  # <-- use the final Dense+Softmax layer
        name="crnn_inference_model"
    )
    return inference_model


def create_text_image(text, img_width, img_height):
    """
    pass
    """
    img = np.ones((img_height, img_width), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (5, img_height // 2), font, 1, (0,), 2, cv2.LINE_AA)
    # cv2.putText(
    #     img,
    #     text,
    #     (2, 24),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.8, (0,),
    #     2,
    #     cv2.LINE_AA
    # )
    img = img.astype("float32") / 255.0  # normalize to [0,1]
    return img


def encode_label(text, char_to_num):
    """
    Simple character to integer mapping
    """
    return [char_to_num[c] for c in text]


def train_crnn(words, img_width, img_height, char_to_num, crnn_model, batch_size):
    """
    test
    """
    images = [create_text_image(w, img_width, img_height) for w in words]

    # expand dims for CNN input
    # x = np.array(images, dtype=np.float32) / 255.0
    x = np.expand_dims(images, -1)  # shape (batch, H, W, 1)

    # Encode labels - simple character to integer mapping
    y = [encode_label(w, char_to_num) for w in words]
    # print(f"\nencoded labels: {y}\n")
    y = tf.ragged.constant(y)
    # print(f"\nencoded labels: {y}\n")

    # make sure all inputs have the same number of samples (100), regardless of batch size. Keras will handle
    # batching internally.
    max_label_length = max(len(seq) for seq in y)
    y_padded = tf.keras.preprocessing.sequence.pad_sequences(
        y.to_list(), maxlen=max_label_length, padding="post"
    )

    # Build inputs with dataset size, not batch size
    num_samples = len(words)

    # Prepare CTC inputs
    # input_lengths = sequence length after CNN downsampling
    # adjust depending on your cnn_model output width
    input_lengths = np.ones((num_samples, 1), dtype=np.int32) * (img_width // 8)
    label_lengths = np.array([[len(w)] for w in words], dtype=np.int32)

    # Train the CRNN model using fit()
    history = crnn_model.fit(
        x={
            "input_layer": x,
            "y_true": y_padded,
            "input_length": input_lengths,
            "label_length": label_lengths
        },
        y=np.zeros((num_samples,)),  # dummy targets
        batch_size=batch_size,
        epochs=100
    )

    return history


def main() -> None:
    # Step 1: Load and pre-process the image
    print("Step 1: Load and pre-process the image\n")
    img_path = "data/input/hello_hershey_simplex.jpg"
    img_width = 128
    img_height = 32
    # Load image and convert to grayscale - text recognition does not need color, it needs only shape, contrast,
    # edges and stroke patterns
    img = load_img(img_path, color_mode="grayscale")
    img_arr = pre_process(img, img_width, img_height)

    # Step 2: Build the CNN model to extract the features
    print("\nStep 2: Build the CNN model to extract the features\n")
    cnn_model = build_cnn_feature_extractor(img_arr)
    cnn_model.summary()

    # Step 3: Use the CNN model to extract the features from the image tensor
    print("\nStep 3: Use the CNN model to extract the features from the image tensor\n")
    features = cnn_model.predict(img_arr)
    print("\nExtracted feature map shape: ", features.shape, "\n")

    # Step 4: Build the RNN (Bi-LSTM) sequence modeling stage for a CRNN OCR model
    print("\nStep 4: Build the RNN (Bi-LSTM) sequence modeling stage for a CRNN OCR model\n")
    rnn_model = build_rnn_sequence_model(cnn_model)
    rnn_model.summary()

    # Step 5: Build the CRNN model with CTC loss
    print("\nStep 5: Build the CRNN model with CTC loss\n")

    with open("data/input/training_words.txt", "r") as f:
        words = [line.strip() for line in f if line.strip()]
    print(f"\nTraining words sample: {words[:10]}\n")

    # for training the CRNN model later in step 8, define the input tensor shape
    alphabet = "abcdefghijklmnopqrstuvwxyz '"
    char_to_num = {c: i for i, c in enumerate(alphabet)}  # 0 reserved for CTC blank
    num_classes = max(char_to_num.values()) + 1  # highest char index + 1
    num_classes += 1  # +1 for CTC blank
    max_label_length = max(len(w) for w in words)
    crnn_model = build_crnn_model(cnn_model, rnn_model, max_label_length, num_classes)
    crnn_model.summary()

    # Step 6: Build the inference model
    print("\nStep 6: Build the inference model\n")
    inference_model = build_inference_model(crnn_model)
    inference_model.summary()

    # Step 7: Compile the CRNN model
    crnn_model.compile(optimizer="adam")

    # Step 8 : Train the CRNN model
    batch_size = 16
    history = train_crnn(words, img_width, img_height, char_to_num, crnn_model, batch_size)

    num_to_char = {v: k for k, v in char_to_num.items()}
    print(f"\nchar_to_num: {char_to_num}\n")
    print(f"\nnum_to_char: {num_to_char}\n")

    def decode_sequence(seq):
        text = ""
        for i in seq:
            if i == -1:  # padding
                continue
            if i == (num_classes - 1):  # blank index
                continue
            text += num_to_char.get(i, "")
        return text

    word = create_text_image("fig", img_width, img_height)
    print(f"\ndecode word: {word.shape}\n")
    word = np.expand_dims(word, axis=-1)
    word = np.expand_dims(word, axis=0)
    print(f"\ndecode word: {word.shape}\n")
    preds = inference_model.predict(word)

    print(f"\nPreds shape: {preds.shape}\n")  # (batch, timesteps, num_classes)
    print(f"\nSample timestep probs: {preds[0, :5, :]}\n")  # first 5 timesteps

    input_length = np.ones(preds.shape[0]) * preds.shape[1]
    decoded, _ = tf.keras.backend.ctc_decode(preds, input_length, greedy=False, beam_width=50)
    decoded_texts = [decode_sequence(seq.numpy()) for seq in decoded[0]]

    print(f"\nInput word image: {word}\n")
    print("Decoded indices:", decoded[0].numpy()[0], "\n")
    print("Decoded text:", decoded_texts[0])


if __name__ == "__main__":
    main()
