"""Pure MicroGPT training and held-out evaluation; no orchestration dependencies."""
import hashlib
import math
from pathlib import Path
import random
import string
import microkernel


def prepare(text, validation_documents=192, split_seed=20261028):
    source_documents = [line.strip() for line in text.splitlines() if line.strip()]
    # Split unique names so an identical name cannot enter training and validation.
    documents = list(dict.fromkeys(source_documents))
    vocabulary = list(string.ascii_lowercase)
    if any(c not in vocabulary for name in documents for c in name):
        raise ValueError("Expected the pinned lowercase a-z names corpus")
    shuffled = list(documents)
    random.Random(split_seed).shuffle(shuffled)
    boundary = int(len(shuffled)*.9)
    training, held_out = shuffled[:boundary], shuffled[boundary:]
    if not 0 < validation_documents <= len(held_out):
        raise ValueError("Validation budget exceeds the held-out split")
    return {"training": training, "validation": held_out[:validation_documents],
        "vocabulary": vocabulary, "split_seed": split_seed,
        "source_documents": len(source_documents), "unique_documents": len(documents),
        "duplicate_documents_removed": len(source_documents)-len(documents), "training_documents": len(training),
        "held_out_documents": len(held_out), "validation_documents": validation_documents,
        "dataset_sha256": hashlib.sha256(text.encode()).hexdigest()}


def train(prepared, configuration):
    checkpoint = microkernel.train(prepared["training"], configuration, prepared["vocabulary"])
    return {"checkpoint": checkpoint, "configuration_id": configuration["configuration_id"],
        "dataset_sha256": prepared["dataset_sha256"], "training_documents": prepared["training_documents"]}


def validate(prepared, trained, shard, shards):
    documents = prepared["validation"]
    quotient, remainder = divmod(len(documents), shards)
    begin = shard*quotient+min(shard,remainder)
    end = begin+quotient+(shard<remainder)
    selected = documents[begin:end]
    if not selected: raise ValueError("Empty validation shard")
    records = microkernel.score(trained["checkpoint"], selected)
    return {"configuration_id": trained["configuration_id"], "shard": shard, "begin": begin,
            "end": end, "records": records}


def aggregate(trained, partitions):
    ordered = sorted(partitions, key=lambda item: item["shard"])
    if [p["shard"] for p in ordered] != list(range(len(ordered))):
        raise ValueError("Validation shard identities are not unique/complete")
    cursor, records = 0, []
    for part in ordered:
        if part["configuration_id"] != trained["configuration_id"] or part["begin"] != cursor:
            raise ValueError("Mismatched model or overlapping/missing validation rows")
        records.extend(part["records"]); cursor = part["end"]
    nll = sum(row["nll_sum"] for row in records)
    tokens = sum(row["tokens"] for row in records)
    checkpoint = trained["checkpoint"]
    return {"configuration_id": trained["configuration_id"], "config": checkpoint["config"],
        "weights_sha256": checkpoint["weights_sha256"], "parameter_count": checkpoint["parameter_count"],
        "loss_history": checkpoint["loss_history"], "train_tokens": checkpoint["train_tokens"],
        "validation_records": records, "validation_tokens": tokens, "validation_nll_sum": nll,
        "validation_nll_per_token": nll/tokens, "validation_perplexity": math.exp(nll/tokens)}


def collect(prepared, evaluations):
    configurations = {r["configuration_id"]: r for r in evaluations}
    if len(configurations) != len(evaluations): raise ValueError("Repeated configuration identity")
    return {"schema": 1, "workload": "microgpt-learning-rate-momentum",
        "data": {k:v for k,v in prepared.items() if k not in ["training","validation"]},
        "configurations": configurations,
        "ranking": sorted(configurations, key=lambda k:(configurations[k]["validation_nll_per_token"],k))}


def read_prepare(path, validation_documents, split_seed):
    return prepare(Path(path).read_text(encoding="utf-8"), validation_documents, split_seed)
