"""Transform module
"""

# Import Library
import tensorflow as tf

# Variabel Global
LABEL_KEY = "label"
FEATURE_KEY = "narasi"


def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"


def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features

    Args:
        inputs: map from feature keys to raw features.

    Return:
        outputs: map from feature keys to transformed features.
    """
    outputs = {}

    # Lowercase the Sentence feature
    text = tf.strings.lower(inputs[FEATURE_KEY])

    # Remove single and double quotes
    text = tf.strings.regex_replace(text, "'", "")
    text = tf.strings.regex_replace(text, '"', "")

    # Remove punctuation
    text = tf.strings.regex_replace(text, r"[^\w\s]", "")

    # Remove extra spaces
    text = tf.strings.regex_replace(text, r"\s+", " ")

    # Trim spaces
    text = tf.strings.strip(text)

    # Assign transformed text to outputs
    outputs[transformed_name(FEATURE_KEY)] = text

    # Cast label to int64
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
