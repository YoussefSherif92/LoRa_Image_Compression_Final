import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ============================================
# SETTINGS
# ============================================
Img_Height = 128
Img_Width = 128
Batch_Size = 16
Epochs = 150
Latent_Dim = 32
Dataset_Path = "Deggendorf_Uniform_Dataset"

# ============================================
# LOAD DATASET
# ============================================
train_dataset = image_dataset_from_directory(
    Dataset_Path,
    color_mode="rgb",
    batch_size=Batch_Size,
    shuffle=True,
    image_size=(Img_Height, Img_Width),
    labels=None,
    validation_split=0.3,      # 30% for testing
    subset="training",         # loads 70%
    seed=42
)

test_dataset = image_dataset_from_directory(
    Dataset_Path,
    color_mode="rgb",
    batch_size=Batch_Size,
    shuffle=False,
    image_size=(Img_Height, Img_Width),
    labels=None,
    validation_split=0.3,
    subset="validation",       # loads 30%
    seed=42
)

# ============================================
# PREPROCESSING: RGB -> HSV
# ============================================
def preprocess_to_hsv(x):
    x = x / 255.0
    x = tf.image.rgb_to_hsv(x)
    return x, x

train_dataset = train_dataset.map(preprocess_to_hsv)
test_dataset = test_dataset.map(preprocess_to_hsv)

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# ============================================
# CUSTOM LOSSES
# ============================================
def pixel_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def edge_loss(y_true, y_pred):
    true_edges = tf.image.sobel_edges(y_true)
    pred_edges = tf.image.sobel_edges(y_pred)
    return tf.reduce_mean(tf.square(true_edges - pred_edges))

def color_loss(y_true, y_pred):
    true_mean = tf.reduce_mean(y_true, axis=[1, 2])
    pred_mean = tf.reduce_mean(y_pred, axis=[1, 2])
    return tf.reduce_mean(tf.square(true_mean - pred_mean))

def structure_loss(y_true, y_pred):
    ssim = tf.reduce_mean(
        tf.image.ssim(
            y_true,
            y_pred,
            max_val=1.0
        )
    )
    return 1.0 - ssim

def combined_loss(y_true, y_pred):
    p_loss = pixel_loss(y_true, y_pred)
    e_loss = edge_loss(y_true, y_pred)
    c_loss = color_loss(y_true, y_pred)
    s_loss = structure_loss(y_true, y_pred)

    return (
        0.40 * p_loss +
        0.25 * e_loss +
        0.15 * c_loss +
        0.20 * s_loss
    )

# ============================================
# BUILD ENCODER
# ============================================
encoder_input = layers.Input(
    shape=(Img_Height, Img_Width, 3),
    name="hsv_encoder_input"
)  # (128,128,3) = 128*128*3 = 49152 input values

x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(encoder_input)     # (128,128,32)
x = layers.MaxPooling2D((2, 2), padding="same")(x)      # (64,64,32)

x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)     # (64,64,64)
x = layers.MaxPooling2D((2, 2), padding="same")(x)      # (32,32,64)

x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)    # (32,32,128)
x = layers.MaxPooling2D((2, 2), padding="same")(x)      # (16,16,128) = 16*16*128 = 32768 final feature values

x = layers.Flatten()(x)     # (32768,)

latent = layers.Dense(Latent_Dim, name="latent_vector")(x)      # (32,)

encoder = Model(encoder_input, latent, name="encoder")

# ============================================
# BUILD DECODER
# ============================================
decoder_input = layers.Input(
    shape=(Latent_Dim,),
    name="decoder_input"
)       # (32,)

x = layers.Dense(16 * 16 * 128, activation="relu")(decoder_input)       # (32768,)

x = layers.Reshape((16, 16, 128))(x)       # (16,16,128)

x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)    # (16,16,128)
x = layers.UpSampling2D((2, 2))(x)      # (32,32,128)

x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)     # (32,32,64)
x = layers.UpSampling2D((2, 2))(x)      # (64,64,64)

x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)     # (64,64,32)
x = layers.UpSampling2D((2, 2))(x)      # (128,128,32)

decoder_output = layers.Conv2D(
    3,
    (3, 3),
    activation="sigmoid",
    padding="same",
    name="hsv_output"
)(x)        # (128,128,3)

decoder = Model(decoder_input, decoder_output, name="decoder")

# ============================================
# BUILD AUTOENCODER
# ============================================
autoencoder_input = encoder_input
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)

autoencoder = Model(autoencoder_input, decoded, name="autoencoder")

# ============================================
# COMPILE
# ============================================
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=combined_loss,
    metrics = [
        pixel_loss,
        edge_loss,
        color_loss,
        structure_loss
    ]
)

autoencoder.summary()

# ============================================
# EARLY STOPPING
# ============================================
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# ============================================
# TRAIN
# ============================================
history = autoencoder.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=Epochs,
    callbacks=[early_stopping]
)

# ============================================
# SAVE MODELS
# ============================================
encoder.save("encoder_model.h5")
decoder.save("decoder_model.h5")
autoencoder.save("autoencoder_model.h5")

# ============================================
# QUANTIZATION TEST
# ============================================
for batch_x, _ in test_dataset.take(1):

    sample = batch_x[:1]

    latent_vector = encoder.predict(sample)
    # latent_vector shape = (1,32)
    # float32 latent size = 32 * 4 bytes = 128 bytes

    latent_min = latent_vector.min()
    latent_max = latent_vector.max()

    latent_uint8 = (
        (latent_vector - latent_min) /
        (latent_max - latent_min + 1e-8) * 255
    ).astype(np.uint8)

    # uint8 latent size = 32 * 1 byte = 32 bytes

    print("Original latent shape:", latent_vector.shape)
    print("Original latent dtype:", latent_vector.dtype)
    print("Original latent size:", latent_vector.nbytes, "bytes")

    print("Quantized latent shape:", latent_uint8.shape)
    print("Quantized latent dtype:", latent_uint8.dtype)
    print("Quantized latent size:", latent_uint8.nbytes, "bytes")

    latent_uint8.tofile("latent_packet_32bytes.bin")

    np.save("latent_min.npy", latent_min)
    np.save("latent_max.npy", latent_max)

    break

# ============================================
# PLOT LOSS
# ============================================
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.grid(True)
plt.legend()
plt.show()
