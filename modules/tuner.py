"""Tuner module
"""

# Import Library
import kerastuner as kt
import tensorflow as tf
from keras.layers import (
    Input,
    Dense,
    Dropout,
    Conv1D,
    GlobalMaxPooling1D,
    Embedding,
    TextVectorization,
)
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult
import tensorflow_transform as tft
from transform import LABEL_KEY, FEATURE_KEY, transformed_name

BATCH_SIZE = 64
TUNE_EPOCHS = 5
MAX_TRIALS = 30
RANDOM_NUMBER = 42


def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=BATCH_SIZE):
    """Generates features and labels for tuning.
    Args:
        file_pattern: input tfrecord file pattern.
        tf_transform_output: A TFTransformOutput.
        batch_size: representing the number of consecutive elements of
        returned dataset to combine in a single batch

    Returns:
        A dataset that contains (features, indices) tuple where features
        is a dictionary of Tensors, and indices is a single Tensor of
        label indices.
    """
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset


def cnn_model(hp, vectorize_layer):
    """Build Keras model
        hp : HyperParameters
        vectorize_layer : TextVectorization layer adapted from Tuner
    Returns:
        A Keras model object"""

    inputs = Input(shape=(1,), dtype=tf.string, name=transformed_name(FEATURE_KEY))
    reshaped_narrative = tf.reshape(inputs, [-1])
    layers = vectorize_layer(reshaped_narrative)
    layers = Embedding(
        input_dim=vectorize_layer.vocabulary_size(),
        output_dim=hp.Int("embedding_dim", min_value=16, max_value=128, step=8),
    )(layers)

    layers = Conv1D(
        filters=hp.Int("conv1d_1", min_value=32, max_value=128, step=16),
        kernel_size=3,
        activation="relu",
    )(layers)

    layers = GlobalMaxPooling1D()(layers)

    layers = Dense(
        units=hp.Int("fc_1", min_value=32, max_value=256, step=32),
        activation="relu",
    )(layers)
    layers = Dropout(hp.Float("dropout_1", min_value=0.0, max_value=0.5, step=0.1))(
        layers
    )

    layers = Dense(
        units=hp.Int("fc_2", min_value=32, max_value=128, step=16),
        activation="relu",
    )(layers)
    layers = Dropout(hp.Float("dropout_2", min_value=0.0, max_value=0.5, step=0.1))(
        layers
    )

    outputs = Dense(1, activation="sigmoid")(layers)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float("learning_rate", 1e-6, 5e-4, sampling="LOG")
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def tuner_fn(fn_args: FnArgs):
    """Builds the tuner component.
    Args:
        fn_args: Holds args used to train the model as name/value pairs.
    Returns:
        A TunerFnResult that consists of the tuner and fit_args
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(
        fn_args.train_files, tf_transform_output, num_epochs=TUNE_EPOCHS
    )
    val_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=TUNE_EPOCHS)

    def build_cnn_model(hp):
        # Create and adapt the vectorize_layer
        vectorize_layer = TextVectorization(
            max_tokens=hp.Int(
                "vocab_size", min_value=10000, max_value=15000, step=1000
            ),
            output_mode="int",
            output_sequence_length=hp.Int(
                "sequence_length", min_value=75, max_value=150, step=25
            ),
        )

        raw_train_dataset = train_set.map(lambda x, y: x[transformed_name(FEATURE_KEY)])
        vectorize_layer.adapt(raw_train_dataset)

        return cnn_model(hp, vectorize_layer)

    tuner = kt.RandomSearch(
        build_cnn_model,
        objective="val_loss",
        seed=RANDOM_NUMBER,
        max_trials=MAX_TRIALS,
        directory=fn_args.working_dir,
        project_name="hoax_tuning",
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={"x": train_set, "validation_data": val_set, "epochs": TUNE_EPOCHS},
    )
