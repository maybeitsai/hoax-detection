""" 
Module for defining and running the hoax detection pipeline using TFX.
"""

import os
from typing import Text

from absl import logging
from tfx.orchestration import metadata, pipeline

PIPELINE_NAME = "hoax-detection"

# pipeline inputs
DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/transform.py"
TUNER_MODULE_FILE = "modules/tuner.py"
TRAINER_MODULE_FILE = "modules/trainer.py"

# pipeline outputs
OUTPUT_BASE = "outputs"
serving_model_dir = os.path.join(OUTPUT_BASE, "serving_model/1")
pipeline_root_dir = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root_dir, "metadata.sqlite")


def init_local_pipeline(components, pipeline_root: Text) -> pipeline.Pipeline:
    """
    Initialize a local TFX pipeline.

    Args:
        components (dict): Dictionary of TFX components.
        pipeline_root (Text): Root directory for the TFX pipeline.

    Returns:
        pipeline.Pipeline: Initialized TFX pipeline object.
    """
    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--direct_running_mode=multi_processing",
        "--direct_num_workers=0",  # Auto-detect based on available CPUs.
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root_dir,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        eam_pipeline_args=beam_args,
    )
