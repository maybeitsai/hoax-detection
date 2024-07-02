"""Initiate tfx pipeline components
"""

import os

import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Tuner,
    Trainer,
    Evaluator,
    Pusher,
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)


def create_example_gen(data_dir):
    """Create CSV ExampleGen component.

    Args:
        data_dir (str): Path to the data directory.

    Returns:
        CsvExampleGen: ExampleGen component.
    """
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ]
        )
    )
    example_gen = CsvExampleGen(input_base=data_dir, output_config=output)
    return example_gen


def create_statistics_schema_validator(example_gen):
    """Create StatisticsGen, SchemaGen, and ExampleValidator components.

    Args:
        example_gen (CsvExampleGen): ExampleGen component.

    Returns:
        tuple: Tuple containing StatisticsGen, SchemaGen, and ExampleValidator components.
    """
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])
    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )
    return statistics_gen, schema_gen, example_validator


def create_transform(example_gen, schema_gen, transform_module):
    """Create Transform component.

    Args:
        example_gen (CsvExampleGen): ExampleGen component.
        schema_gen (SchemaGen): SchemaGen component.
        transform_module (str): Path to the transform module file.

    Returns:
        Transform: Transform component.
    """
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.abspath(transform_module),
    )
    return transform


def create_tuner_trainer(transform, schema_gen, tuning_module, training_module):
    """Create Tuner and Trainer components.

    Args:
        transform (Transform): Transform component.
        schema_gen (SchemaGen): SchemaGen component.
        tuning_module (str): Path to the tuning module file.
        training_module (str): Path to the training module file.

    Returns:
        tuple: Tuple containing Tuner and Trainer components.
    """
    tuner = Tuner(
        module_file=os.path.abspath(tuning_module),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(splits=["train"]),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"]),
    )

    trainer = Trainer(
        module_file=os.path.abspath(training_module),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(splits=["train"]),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"]),
        hyperparameters=tuner.outputs["best_hyperparameters"],
    )
    return tuner, trainer


def create_model_resolver_evaluator(example_gen, trainer):
    """Create Model Resolver and Evaluator components.

    Args:
        example_gen (CsvExampleGen): ExampleGen component.
        trainer (Trainer): Trainer component.

    Returns:
        tuple: Tuple containing Model Resolver and Evaluator components.
    """
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("Latest_blessed_model_resolver")

    slicing_specs = [tfma.SlicingSpec()]

    metrics_specs = [
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name="ExampleCount"),
                tfma.MetricConfig(class_name="AUC"),
                tfma.MetricConfig(class_name="Precision"),
                tfma.MetricConfig(class_name="Recall"),
                tfma.MetricConfig(class_name="ExampleCount"),
                tfma.MetricConfig(
                    class_name="BinaryAccuracy",
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={"value": 0.5}
                        ),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={"value": 0.0001},
                        ),
                    ),
                ),
            ]
        )
    ]

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="label")],
        slicing_specs=slicing_specs,
        metrics_specs=metrics_specs,
    )

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )
    return model_resolver, evaluator


def create_pusher(trainer, evaluator, serving_model_dir):
    """Create Pusher component.

    Args:
        trainer (Trainer): Trainer component.
        evaluator (Evaluator): Evaluator component.
        serving_model_dir (str): Path to the serving model directory.

    Returns:
        Pusher: Pusher component.
    """
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )
    return pusher


def init_components(
    data_dir,
    transform_module,
    tuning_module,
    training_module,
    serving_model_dir,
):
    """Initiate tfx pipeline components

    Args:
        data_dir (str): Path to the data directory.
        transform_module (str): Path to the transform module file.
        tuning_module (str): Path to the tuning module file.
        training_module (str): Path to the training module file.
        serving_model_dir (str): Path to the serving model directory.

    Returns:
        tuple: Tuple containing all TFX pipeline components.
    """
    example_gen = create_example_gen(data_dir)
    statistics_gen, schema_gen, example_validator = create_statistics_schema_validator(
        example_gen
    )
    transform = create_transform(example_gen, schema_gen, transform_module)
    tuner, trainer = create_tuner_trainer(
        transform, schema_gen, tuning_module, training_module
    )
    model_resolver, evaluator = create_model_resolver_evaluator(example_gen, trainer)
    pusher = create_pusher(trainer, evaluator, serving_model_dir)

    return (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    )
