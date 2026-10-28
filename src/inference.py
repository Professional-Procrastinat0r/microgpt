"""List exported checkpoints or restore weights and generate names."""
import argparse
import ast
from pathlib import Path
import random

import checkpoints
import microkernel

PROJECT = Path(__file__).resolve().parents[1]


def generate(checkpoint, samples=20, temperature=0.5, seed=730):
    if samples < 1 or not 0 < temperature <= 1:
        raise ValueError("Samples must be positive and temperature in (0, 1]")
    ns = microkernel.restore(checkpoints.validate_checkpoint(checkpoint))
    ns.update(random=random.Random(seed), temperature=temperature, samples=[])
    # The checksum-verified original sampling loop, with returned output.
    microkernel.upstream_parts()
    tree = ast.parse(Path(microkernel.__file__).with_name("vendor_microgpt.py").read_text(encoding="utf-8"))
    loop = next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "sample_idx")
    loop.iter = ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[ast.Constant(samples)], keywords=[])
    loop.body = [n for n in loop.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Name) and n.value.func.id == "print")]
    loop.body.extend(ast.parse("samples.append(''.join(sample))").body)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), "upstream-sampling", "exec"), ns)
    return ns["samples"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute", type=Path, default=PROJECT/"configs/compute_config.yaml")
    parser.add_argument("--list", action="store_true", help="List checkpoint URIs and training configurations")
    parser.add_argument("--checkpoint", help="Exported JSON path or vfs://artefact/models/<digest>.json")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=730)
    args = parser.parse_args()
    if args.list:
        from expops.storage import VFS
        with VFS.from_config(args.compute, base_path=PROJECT, project_id="microgpt") as fs:
            paths = sorted(fs.glob("artefact/models", "*.json")) if fs.exists("artefact/models") else []
            for path in paths:
                checkpoint = checkpoints.load_bundle(fs.read_bytes(path))
                print(checkpoint["config"]["configuration_id"], "vfs://artefact/models/"+path.name)
            if not paths:
                print("No exported models yet. Run python run.py first.")
        return
    if not args.checkpoint:
        parser.error("Supply --list or --checkpoint")
    if args.checkpoint.startswith("vfs://"):
        from expops.storage import VFS
        with VFS.from_config(args.compute, base_path=PROJECT, project_id="microgpt") as fs:
            raw = fs.read_bytes(args.checkpoint)
    else:
        raw = Path(args.checkpoint).read_bytes()
    checkpoint = checkpoints.load_bundle(raw)
    for name in generate(checkpoint, args.samples, args.temperature, args.seed):
        print(name)


if __name__ == "__main__":
    main()
