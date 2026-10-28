# MicroGPT with ExpOps

A standalone ExpOps example using Andrej Karpathy's MicroGPT to train a small
character-level transformer on names, compare optimizer settings and score
held-out documents in parallel. This folder contains no benchmark harness and
does not import anything from `fyp-report-notes` or `credit-card-fraud`.

## Install and run

Keep this folder beside the **current** `expops-platform` checkout. From this
folder, using Python 3.12 or newer (validated with Python 3.14.3):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py
```

The requirements install the sibling platform in editable mode. Use the latest
local implementation with the cache fixes, not predecessor `main`. If you already
have an environment with that ExpOps checkout installed, simply use its Python to
run `run.py`; there is no need for another environment. On Linux/macOS activate
with `source .venv/bin/activate` instead.

`run.py` calls ExpOps' public `MLPlatform.run_pipeline_for_project` API with the
native project/compute YAML. The project declares `system: {}` and uses the
invoking environment: MicroGPT's numerical code needs only the standard library,
while the requirements supply ExpOps and its execution dependencies. This example
does not demonstrate multiple dependency environments. The launcher supplies
project-local scratch/cache defaults and preserves explicit caller overrides.
The ordinary ExpOps CLI can also consume these configurations; it has its own
environment preparation and user-home interpreter-marker files.

### Run training through the ExpOps CLI

After installing the requirements, run the **training experiment** directly
through ExpOps from this project's directory:

```powershell
Set-Location D:\NUS\FYP\microgpt
.venv\Scripts\Activate.ps1
expops run configs/compute_config.yaml --local
```

If you are using an existing environment, activate that environment instead.
The equivalent command when `expops` is not on PATH is:

```powershell
python -m expops.main run configs/compute_config.yaml --local
```

ExpOps retrieves the project through the `code` mount, reads
`configs/project_config.yaml`, starts the six local workers, trains all six model
configurations, evaluates their held-out shards and exports the models/report.
These commands perform training; `src/inference.py` only loads an already saved
checkpoint. `python run.py` is the public-API alternative for the same project
and compute configuration, with project-local scratch defaults.

Repeat a training command unchanged to reuse cached stages. Changing a model's
parameters retrains the affected branch and its downstream evaluation stages;
changes to shared source/dependencies can invalidate the full experiment.

The dataset is already included in [data/names.txt](data/names.txt). Training does
not download data, call external services or require cloud credentials. Six local
worker processes execute the default experiment. Change `num_workers` in
[configs/compute_config.yaml](configs/compute_config.yaml) to fit your machine.

## Experiment and configuration

- Six learning-rate/momentum combinations: `0.003/0.01/0.03` crossed with
  Adam beta1 `0.85/0.90`; 300 optimizer updates each.
- One initialization/training-order seed, 42. There is no seed fan-out.
- A one-layer, 16-wide transformer with four attention heads, context length 16
  and 4,192 trainable parameters.
- Remove duplicate names before the fixed 90/10 split: 26,544 training names
  and 2,950 held-out names. The default scores a fixed 192-name held-out subset.
- Three disjoint scoring shards per model; aggregate by total token count.

Independent configurations provide **task parallelism**; independent validation
documents provide **data parallelism**. Training steps within each model stay
sequential. The 32-stage explicit graph is prepare -> six training branches ->
18 scoring branches -> six summaries -> collection. It uses ordinary native
ExpOps process dependencies, rather than native seed/data expansion declarations.

Edit [configs/project_config.yaml](configs/project_config.yaml) to change the
experiment. Common model fields use the YAML anchor `&model_defaults` in the first
training process and are inherited by the other five configurations. Change
`num_steps` there once to change all training budgets. Each branch overrides its
learning rate, momentum and descriptive ID. The `prepare` parameters control the
validation budget/split. To change the number of validation shards, update the
explicit scoring nodes, dependency edges and aggregation `part_keys` together.

The input is streamed through `context.inputs`; model/report exports use
`context.fs`. Storage roots and compute capacity live in the compute YAML.
The application does not embed native storage paths in worker code.

## Find and load trained weights

**Every training process explicitly saves an inference checkpoint to the
`artefact` mount**, in addition to its normal cached process result:

```text
vfs://artefact/models/<SHA256-of-complete-JSON>.json
```

With the supplied local mount this resolves to:

```text
microgpt/.microgpt/storage/artefact/models/<digest>.json
```

List the saved models and their configuration IDs:

```powershell
python src/inference.py --list
```

Copy one URI from the listing and generate names:

```powershell
python src/inference.py --checkpoint vfs://artefact/models/REPLACE_WITH_DIGEST.json --samples 20
```

An actual file from the first validated run is:

```powershell
python src/inference.py --checkpoint vfs://artefact/models/6ef61526d89216612a6d4bcca0f80878583bb1a9b4feaa002006174d319f746b.json --samples 20
```

This is configuration `lr0100_b85` at the checked-in defaults. The digest changes
when the complete checkpoint changes; `--list` is the general discovery command.
A native downloaded/copied JSON path also works:

```powershell
python src/inference.py --checkpoint .microgpt/storage/artefact/models/6ef61526d89216612a6d4bcca0f80878583bb1a9b4feaa002006174d319f746b.json
```

For loading in your own Python code, with `src` on your import path:

```python
from pathlib import Path
from checkpoints import load_bundle
from microkernel import restore

checkpoint = load_bundle(Path("saved-checkpoint.json").read_bytes())
model = restore(checkpoint)
# model['state_dict'] contains restored Value objects; model['gpt'] is the forward function.
```

The JSON includes floating-point weights, model/optimizer settings, vocabulary,
training-loss history, dataset hash, original kernel hash and weight hash.
Loading checks the kernel identity, weight digest and matrix layout. Training
exports use atomic VFS writes and read-back verification. Reports contain each
checkpoint's URI, byte size and full-file SHA256. These are application-owned
exports and are not automatically registered in the platform's chart index.

**These are inference checkpoints, not exact training-resume checkpoints.** Adam
moments, optimizer progress and training RNG state are not exported. Initializing
further training from these weights would require an explicit warm-start training
path; the supplied `train` function always initializes a new model.

## Where other outputs go

| Output | Default location relative to this folder |
| --- | --- |
| Explicit model JSON | `.microgpt/storage/artefact/models/` |
| Full comparison report JSON | `.microgpt/storage/artefact/reports/` |
| Intermediate results/checkpoints for reuse | `.microgpt/storage/cache/` |
| SQLite run/metric/cache-index metadata | `.microgpt/storage/metadata.db` |
| Platform logs | `.microgpt/logs/` |
| Temporary workspaces | `.microgpt/workspaces/` |

The final report ranks the six configurations by validation NLL and includes
loss traces, per-document scoring, perplexities and model export references.
Process logs print the report and model URIs. Files in `reports/` have
content-derived names; no mutable `latest` selection is made automatically.

An unchanged repeat can reuse the cached stages. Export side effects do not run
again on a cache hit, so retain the artefact files. Deleting an export does not
automatically make its cached training process rerun. Cache reuse has a 720-hour
TTL in this profile; the application adds no expiry rule to explicit model
exports. Retention of exported files remains your responsibility.

You can replace the artefact mount with a supported remote provider without
changing the process functions. For `inference.py`, pass the corresponding local
compute file with `--compute configs/your-compute.yaml` and supply that provider's
dependencies/credentials. A `vfs://` URI is relative to its configured mount table;
it is not a globally resolvable download URL. Native JSON-file inference needs no
ExpOps import and can run using the standard library with this `src` directory.

## Attribution and scope

- [Karpathy's original source](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95),
  pinned revision `14fb038816c7aae0bb9342c2dbf1a51dd134a5ff`.
- [Original names dataset](https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt).
- [Source and input checksums](source-lock.json).

[src/vendor_microgpt.py](src/vendor_microgpt.py) is the unchanged original file.
It is never imported as a module: [microkernel.py](src/microkernel.py) executes
checksum-verified numerical AST sections, excluding the original downloader and
top-level execution. The adapter exposes independent model/RNG state, parameters,
weights and training history. The benchmark's numerical parity test verified the
adapted training loop against the original; this copy retains the same kernel.
The standalone example adds model exports and a loading/generation command.

This is a small educational CPU workload, not distributed-gradient/GPU training
or a production language model. Hyperparameter conclusions are conditional on the
short training budget, fixed initialization and held-out subset.

## Checks

```powershell
python -m unittest discover -s tests -v
```

Four checks cover export/reload scoring and generation, damaged weights, wrong
matrix layouts and kernel mismatches. A complete default run was also verified:
32 processes completed, six JSON checkpoints and one report were exported, and
the scientific outputs match the earlier benchmark. Validation details are in
[VALIDATION.md](VALIDATION.md).
