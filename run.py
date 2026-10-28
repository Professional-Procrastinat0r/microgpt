"""Run this project's native ExpOps pipeline in the installed environment."""
from pathlib import Path
import os


if __name__ == "__main__":
    project = Path(__file__).resolve().parent
    # Keep writable runtime state alongside this example. Explicit caller
    # settings still win; no machine-specific paths are committed.
    paths = {"XDG_CACHE_HOME": "cache", "PIP_CACHE_DIR": "pip-cache",
             "MLOPS_WORKSPACE_BASE_DIR": "workspaces", "DASK_TEMPORARY_DIRECTORY": "dask",
             "MLOPS_DATA_MATERIALIZATION_DIR": "materializations", "JOBLIB_TEMP_FOLDER": "joblib",
             "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp"}
    for name, directory in paths.items():
        destination = project/".microgpt"/directory
        destination.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(destination))
    os.environ.setdefault("MLOPS_WORKSPACE_DIR", str(project.parent))
    os.environ.setdefault("MLOPS_PROJECT_ID", "microgpt")
    os.chdir(project.parent)
    from expops.core.compute_config import resolve_compute_plan
    from expops.platform import MLPlatform
    result = MLPlatform().run_pipeline_for_project(
        "microgpt", str(project/"configs/project_config.yaml"),
        compute_plan=resolve_compute_plan(project/"configs/compute_config.yaml"))
    print(f"Completed run: {result['run_id']}")
    print("List exported checkpoints: python src/inference.py --list")
