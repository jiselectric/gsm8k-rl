"""Run many eval configs from a YAML file; each model is loaded once.

    python src/sweep.py configs/phase1.yaml             # run everything (resumable)
    python src/sweep.py configs/phase1.yaml --dry-run   # list runs, run nothing
    python src/sweep.py configs/phase1.yaml --n 20      # quick version of the sweep

Finished runs (results/<split>/<model>__<tag>.jsonl exists) are skipped, so you
can Ctrl+C anytime and rerun the same command to continue. Then:

    python src/aggregate.py
"""
import argparse
import gc
import itertools
from collections import defaultdict

import mlx.core as mx
import yaml
from eval import load_model, result_path, run_eval


def expand(spec: dict) -> list[tuple[str, dict]]:
    """Experiment -> [(tag, overrides)]. List values expand into a grid."""
    overrides = spec.get("set") or {}
    grid_keys = [k for k, v in overrides.items() if isinstance(v, list)]
    fixed = {k: v for k, v in overrides.items() if k not in grid_keys}
    if not grid_keys:
        return [(spec["name"], fixed)]
    runs = []
    for values in itertools.product(*(overrides[k] for k in grid_keys)):
        combo = dict(zip(grid_keys, values))
        tag = spec["name"] + "".join(f"_{k}{v}" for k, v in combo.items())
        runs.append((tag, {**fixed, **combo}))
    return runs


def plan(cfg_file: dict, n_override: int | None) -> dict[str, list[tuple[str, dict]]]:
    """model -> [(tag, full eval config)] for every run in the sweep."""
    runs = defaultdict(list)
    for spec in cfg_file["experiments"]:
        for model in spec.get("models", cfg_file["models"]):
            for tag, overrides in expand(spec):
                cfg = {"model": model, "split": cfg_file.get("split", "dev"),
                       "n": n_override if n_override is not None else cfg_file.get("n"),
                       **cfg_file["baseline"], **overrides}
                if n_override is not None:
                    tag += f"_n{n_override}"
                runs[model].append((tag, cfg))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n", type=int, default=None, help="override n (quick sweep)")
    args = ap.parse_args()

    cfg_file = yaml.safe_load(open(args.config))
    revisions = cfg_file.get("revisions") or {}
    runs = plan(cfg_file, args.n)
    todo = {m: [(t, c) for t, c in rs if not result_path(c, t).exists()]
            for m, rs in runs.items()}
    total = sum(len(rs) for rs in runs.values())
    left = sum(len(rs) for rs in todo.values())
    print(f"{total} runs in sweep, {total - left} already done, {left} to run\n")
    for model, rs in runs.items():
        for tag, cfg in rs:
            done = "done" if result_path(cfg, tag).exists() else "todo"
            print(f"  [{done}] {model:36s} {tag}")
    if args.dry_run or not left:
        return

    for model, rs in todo.items():
        if not rs:
            continue
        print(f"\n=== loading {model} ({len(rs)} runs)")
        mdl, tok, rev = load_model(model, revisions.get(model))
        print(f"    revision {rev}")
        for tag, cfg in rs:
            cfg["revision"] = rev
            n_correct, n = run_eval(mdl, tok, cfg, result_path(cfg, tag))
            print(f"    {tag}: {n_correct}/{n} = {100 * n_correct / n:.1f}%")
        del mdl, tok          # free memory before the next model
        gc.collect()
        mx.clear_cache()
    print("\nSweep finished. Next: python src/aggregate.py")


if __name__ == "__main__":
    main()
