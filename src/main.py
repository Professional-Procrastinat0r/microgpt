"""ExpOps processes for a MicroGPT optimizer sweep and partitioned validation."""
from expops.core import log_metric, process

import checkpoints
import workload


@process()
def prepare(validation_documents=192, split_seed=20261028, context=None):
    with context.inputs["names"].open("r", encoding="utf-8") as source:
        prepared = workload.prepare(source.read(), validation_documents, split_seed)
    return {"prepared": prepared}


@process()
def train(prepared, configuration, output_key, context=None):
    trained = workload.train(prepared, configuration)
    trained["model_artifact"] = checkpoints.export_model(context.fs, trained)
    log_metric("final_training_loss", trained["checkpoint"]["loss_history"][-1])
    print(f"Model {trained['configuration_id']}: {trained['model_artifact']['uri']}")
    return {output_key: trained}


@process()
def validate(prepared, trained_key, shard, shards, output_key, context=None, **kwargs):
    return {output_key: workload.validate(prepared, kwargs[trained_key], shard, shards)}


@process()
def aggregate(trained_key, part_keys, output_key, context=None, **kwargs):
    trained = kwargs[trained_key]
    result = workload.aggregate(trained, [kwargs[key] for key in part_keys])
    result["model_artifact"] = trained["model_artifact"]
    log_metric("validation_nll_per_token", result["validation_nll_per_token"])
    log_metric("validation_perplexity", result["validation_perplexity"])
    return {output_key: result}


@process()
def collect(prepared, result_keys, context=None, **kwargs):
    report = workload.collect(prepared, [kwargs[key] for key in result_keys])
    artifact = checkpoints.publish_json(context.fs, "reports", report)
    print(f"Experiment report: {artifact['uri']}")
    print(f"Lowest validation loss: {report['ranking'][0]}")
    return {"report": report, "report_artifact": artifact}
