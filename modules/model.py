""" 
Module for Make Keras Model
"""

import tensorflow as tf
from keras.layers import (
    Input,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
)
from transform import transformed_name, FEATURE_KEY


def cnn_model(hp, vectorize_layer):
    """Build Keras model
    Args:
        hp : HyperParameters
        vectorize_layer : TextVectorization layer adapted from Tuner
    Returns:
        A Keras model object
    """
    inputs = Input(shape=(1,), dtype=tf.string, name=transformed_name(FEATURE_KEY))
    reshaped_narrative = tf.reshape(inputs, [-1])
    layers = vectorize_layer(reshaped_narrative)
    layers = Embedding(
        input_dim=vectorize_layer.vocabulary_size(),
        output_dim=(
            hp["embedding_dim"]
            if isinstance(hp, dict)
            else hp.Int("embedding_dim", min_value=16, max_value=128, step=8)
        ),
    )(layers)

    layers = Conv1D(
        filters=(
            hp["conv1d_1"]
            if isinstance(hp, dict)
            else hp.Int("conv1d_1", min_value=32, max_value=128, step=16)
        ),
        kernel_size=3,
        activation="relu",
    )(layers)
    layers = GlobalMaxPooling1D()(layers)

    layers = Dense(
        units=(
            hp["fc_1"]
            if isinstance(hp, dict)
            else hp.Int("fc_1", min_value=32, max_value=256, step=32)
        ),
        activation="relu",
    )(layers)
    layers = Dropout(
        hp["dropout_1"]
        if isinstance(hp, dict)
        else hp.Float("dropout_1", min_value=0.0, max_value=0.5, step=0.1)
    )(layers)
    layers = Dense(
        units=(
            hp["fc_2"]
            if isinstance(hp, dict)
            else hp.Int("fc_2", min_value=32, max_value=128, step=16)
        ),
        activation="relu",
    )(layers)
    layers = Dropout(
        hp["dropout_2"]
        if isinstance(hp, dict)
        else hp.Float("dropout_2", min_value=0.0, max_value=0.5, step=0.1)
    )(layers)
    outputs = Dense(1, activation="sigmoid")(layers)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp["learning_rate"]
            if isinstance(hp, dict)
            else hp.Float("learning_rate", 1e-6, 5e-4, sampling="LOG")
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
