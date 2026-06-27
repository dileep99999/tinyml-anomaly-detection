import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cwru_autoencoder(input_length=2048, bottleneck=32):
    inp = keras.Input(shape=(input_length, 1))

    # encoder
    x = layers.Conv1D(16, 7, strides=2, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, 5, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(16, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    shape_before = x.shape[1:]
    x = layers.Flatten()(x)
    x = layers.Dense(bottleneck, activation="relu")(x)  # bottleneck

    # decoder
    x = layers.Dense(shape_before[0] * shape_before[1], activation="relu")(x)
    x = layers.Reshape(shape_before)(x)
    x = layers.Conv1DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1DTranspose(32, 5, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1DTranspose(16, 7, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv1D(1, 1, padding="same", activation="linear")(x)

    # crop back to exact input size
    x = layers.Cropping1D((0, x.shape[1] - input_length))(x)

    model = keras.Model(inputs=inp, outputs=x, name="CWRU_CAE")
    return model


def build_mimii_autoencoder(input_shape=(64, 313, 1), bottleneck=64):
    inp = keras.Input(shape=input_shape)

    # encoder
    x = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    shape_before = x.shape[1:]
    x = layers.Flatten()(x)
    x = layers.Dense(bottleneck, activation="relu")(x)  # bottleneck

    # decoder
    x = layers.Dense(shape_before[0] * shape_before[1] * shape_before[2], activation="relu")(x)
    x = layers.Reshape(shape_before)(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(1, 1, padding="same", activation="linear")(x)

    x = layers.Cropping2D((
        (0, x.shape[1] - input_shape[0]),
        (0, x.shape[2] - input_shape[1])
    ))(x)

    model = keras.Model(inputs=inp, outputs=x, name="MIMII_CAE")
    return model


if __name__ == "__main__":
    m = build_cwru_autoencoder()
    m.summary()
    total = sum(tf.size(w).numpy() for w in m.trainable_weights)
    print(f"Parameters: {total:,}  (~{total*4/1024:.0f} KB float32)")
