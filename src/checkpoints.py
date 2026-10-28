"""Explicit JSON model exports, separate from ExpOps' intermediate-result cache."""
import hashlib
import json
import math

import microkernel


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def validate_checkpoint(checkpoint):
    """Reject damaged weights and incompatible layouts before inference."""
    weights = checkpoint["weights"]
    if hashlib.sha256(canonical_bytes(weights)).hexdigest() != checkpoint["weights_sha256"]:
        raise ValueError("Checkpoint weight checksum differs")
    vocabulary = checkpoint["vocabulary"]
    if not vocabulary or len(set(vocabulary)) != len(vocabulary) or any(len(c) != 1 for c in vocabulary):
        raise ValueError("Invalid checkpoint vocabulary")
    ns = microkernel.model_namespace(checkpoint["config"], vocabulary)
    exec(microkernel.upstream_parts()[1], ns)
    expected = ns["state_dict"]
    if set(weights) != set(expected):
        raise ValueError("Checkpoint parameter names differ from model architecture")
    count = 0
    for key, matrix in weights.items():
        shape = expected[key]
        if len(matrix) != len(shape):
            raise ValueError(f"Invalid row count for {key}")
        for row, expected_row in zip(matrix, shape):
            if len(row) != len(expected_row) or any(type(x) not in (int, float) or not math.isfinite(x) for x in row):
                raise ValueError(f"Invalid weight row for {key}")
            count += len(row)
    if count != checkpoint["parameter_count"]:
        raise ValueError("Checkpoint parameter count differs")
    return checkpoint


def publish_json(fs, directory, value):
    """Content-address an export and verify it through the same mount."""
    raw = canonical_bytes(value)
    digest = hashlib.sha256(raw).hexdigest()
    uri = f"vfs://artefact/{directory}/{digest}.json"
    fs.write_bytes(uri, raw)
    if fs.read_bytes(uri) != raw:
        raise ValueError("Export failed read-back verification")
    return {"uri": uri, "sha256": digest, "size_bytes": len(raw)}


def export_model(fs, trained):
    checkpoint = validate_checkpoint(trained["checkpoint"])
    bundle = {"schema": 1, "kind": "microgpt-inference-checkpoint",
              "upstream_source_sha256": microkernel.VENDOR_SHA256,
              "dataset_sha256": trained["dataset_sha256"], "checkpoint": checkpoint}
    # The directory has no configuration-controlled path components.
    return publish_json(fs, "models", bundle)


def load_bundle(raw):
    bundle = json.loads(raw)
    if bundle.get("schema") != 1 or bundle.get("kind") != "microgpt-inference-checkpoint":
        raise ValueError("Expected an exported MicroGPT inference checkpoint")
    if bundle.get("upstream_source_sha256") != microkernel.VENDOR_SHA256:
        raise ValueError("Checkpoint requires a different upstream numerical kernel")
    return validate_checkpoint(bundle["checkpoint"])
