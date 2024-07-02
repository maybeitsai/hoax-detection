"""Training module 
"""

import os
import tensorflow as tf
from keras.callbacks import (
    EarlyStopping,
    TensorBoard,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from keras.utils.vis_utils import plot_model
from keras.layers import TextVectorization
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from transform import LABEL_KEY, FEATURE_KEY, transformed_name
from tuner import input_fn, BATCH_SIZE
from model import cnn_model

TRAIN_EPOCHS = 30


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs):
    """Train the model based on given args.
    Args:
    fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(
        fn_args.train_files, tf_transform_output, num_epochs=TRAIN_EPOCHS
    )
    val_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=TRAIN_EPOCHS)

    print(fn_args.hyperparameters)

    # Parse hyperparameters directly
    hyperparameters = fn_args.hyperparameters
    raw_train_dataset = train_set.map(lambda x, y: x[transformed_name(FEATURE_KEY)])
    vectorize_layer = TextVectorization(
        max_tokens=hyperparameters["values"]["vocab_size"],
        output_mode="int",
        output_sequence_length=hyperparameters["values"]["sequence_length"],
    )

    vectorize_layer.adapt(raw_train_dataset)

    model = cnn_model(hyperparameters["values"], vectorize_layer)

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = TensorBoard(log_dir=log_dir, update_freq="batch")
    early_stop = EarlyStopping(monitor="val_loss", verbose=1, patience=3)
    check_point = ModelCheckpoint(
        fn_args.serving_model_dir, monitor="val_loss", verbose=1, save_best_only=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.3, patience=1, min_lr=1e-7, verbose=1
    )

    model.fit(
        train_set,
        batch_size=BATCH_SIZE,
        validation_data=val_set,
        callbacks=[tensorboard_callback, early_stop, check_point, reduce_lr],
        epochs=TRAIN_EPOCHS,
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        )
    }
    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)

    plot_model(
        model, to_file="images/model_plot.png", show_shapes=True, show_layer_names=True
    )
