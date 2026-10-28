"""Execute the pinned upstream numerical AST with isolated per-model state.

The vendor file is never imported: its downloader and top-level training/sample
side effects cannot run. Value and forward functions, parameter initialization
and the Adam training loop come directly from its checked original AST.
"""
from __future__ import annotations
import ast
import functools
import hashlib
import json
import math
from pathlib import Path
import random

VENDOR_SHA256 = "d47d88c2fd432c8ebdc1048beab7f7eb64ea7e0e664e11b812d72a6d95ebccee"


@functools.lru_cache(maxsize=1)
def upstream_parts():
    path = Path(__file__).with_name("vendor_microgpt.py")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != VENDOR_SHA256:
        raise ValueError("Pinned upstream source checksum differs")
    tree = ast.parse(raw.decode("utf-8"), filename=str(path))
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assert {n.name for n in definitions} == {"Value", "linear", "softmax", "rmsnorm", "gpt"}
    def target(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    begin = next(i for i, n in enumerate(tree.body) if target(n, "matrix"))
    end = next(i for i, n in enumerate(tree.body) if target(n, "params"))
    initializer = tree.body[begin:end+1]
    loop = next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "step")
    # Only the progress print is replaced by a numerical loss history append.
    loop.body = [n for n in loop.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Name) and n.value.func.id == "print")]
    loop.body.extend(ast.parse("history.append(loss.data)\ntrain_tokens += n").body)
    def compile_nodes(nodes):
        return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec")
    return compile_nodes(definitions), compile_nodes(initializer), compile_nodes([loop])


def model_namespace(config, vocabulary):
    if config["n_embd"] % config["n_head"]:
        raise ValueError("Embedding width must divide into attention heads")
    ns = {"__name__": "isolated_microgpt", "math": math, "random": random.Random(config["seed"]),
          **config, "uchars": vocabulary, "BOS": len(vocabulary), "vocab_size": len(vocabulary)+1,
          "head_dim": config["n_embd"] // config["n_head"]}
    exec(upstream_parts()[0], ns)
    return ns


def train(documents, config, vocabulary):
    ns = model_namespace(config, vocabulary)
    ns["docs"] = list(documents)
    # Preserve the upstream RNG order: document shuffle before Gaussian weights.
    ns["random"].shuffle(ns["docs"])
    exec(upstream_parts()[1], ns)
    ns.update({"m": [0.0]*len(ns["params"]), "v": [0.0]*len(ns["params"]),
               "history": [], "train_tokens": 0})
    exec(upstream_parts()[2], ns)
    weights = {key: [[value.data for value in row] for row in matrix]
               for key, matrix in ns["state_dict"].items()}
    if not all(math.isfinite(x) for matrix in weights.values() for row in matrix for x in row):
        raise ValueError("Training produced non-finite weights")
    return {"config": dict(config), "vocabulary": list(vocabulary), "weights": weights,
            "loss_history": ns["history"], "train_tokens": ns["train_tokens"],
            "parameter_count": len(ns["params"]),
            "weights_sha256": hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()}


def restore(checkpoint):
    ns = model_namespace(checkpoint["config"], checkpoint["vocabulary"])
    value_class = ns["Value"]
    ns["state_dict"] = {key: [[value_class(x) for x in row] for row in matrix]
                        for key, matrix in checkpoint["weights"].items()}
    return ns


def score(checkpoint, documents):
    ns = restore(checkpoint)
    encode = {char: i for i, char in enumerate(ns["uchars"])}
    result = []
    for document in documents:
        tokens = [ns["BOS"], *[encode[c] for c in document], ns["BOS"]]
        count = min(ns["block_size"], len(tokens)-1)
        keys, values = [[] for _ in range(ns["n_layer"])], [[] for _ in range(ns["n_layer"])]
        losses = []
        for position in range(count):
            logits = ns["gpt"](tokens[position], position, keys, values)
            probs = ns["softmax"](logits)
            losses.append(-probs[tokens[position+1]].log().data)
        result.append({"document": document, "tokens": count, "nll_sum": sum(losses)})
    return result
