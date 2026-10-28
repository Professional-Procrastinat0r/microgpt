# Standalone example validation

Verified on 28 October 2026, Windows, Python 3.14.3, using the current local
ExpOps checkout and the preinstalled fraud-model environment. No changes were
made to production platform code or the recorded benchmark experiments.

| Check | Result |
| --- | --- |
| Full `python run.py` at checked-in defaults | 32 executed, 0 failed |
| Unchanged repeat | 32 cached, 0 executed, 0 failed |
| Native CLI `python -m expops.main run configs/compute_config.yaml --local` | Exit 0; six workers; 32 cached, 0 failed |
| Model exports | Six content-addressed JSON files, read-back verified |
| Report export | One JSON report with references to all six model files |
| Comparison with benchmark's complete scientific report | Exact equality after removing the added model export references |
| Checkpoint listing command | Six configuration IDs/URIs |
| Generation from default model's exported VFS URI | `bevanl`, `anira`, `ramin`, `mashon`, `latan` at defaults, five samples |
| Four export/restore/corruption tests | Passed |

Full run ID: `project-microgpt-20261028130248-b84f4b10`.
Cached repeat: `project-microgpt-20261028130633-fdad4bf7`.
Native CLI check: `project-microgpt-20261028133538-485ee6d7`.
Raw platform logs and outputs remain in the ignored `.microgpt/` folder.
These integration checks are not new benchmark timings.

The full run created:

- Default model: `vfs://artefact/models/6ef61526d89216612a6d4bcca0f80878583bb1a9b4feaa002006174d319f746b.json`.
- Report: `vfs://artefact/reports/913a49ad35b6a0becb44eeb5633af3652e48876e5cfb9a14d98d8972226e1b1f.json`.

The supplied launcher uses the public platform API with the declared system
environment and project-local runtime defaults. The native CLI path was also
subsequently verified, with access to its user-home interpreter marker. Neither
path replaces any executor, cache, storage or process implementation.
