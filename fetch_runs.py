"""
fetch_runs.py  —  DA6401 Assignment-3 W&B Data Fetcher
Run: python fetch_runs.py --project da6401-assignment-3 --entity YOUR_ENTITY
Output: wandb_runs_data.json  (share this file with Claude)
"""

import argparse
import json
import wandb

EXPECTED_GROUPS = [
    "noam_vs_fixed_lr",
    "scaling_factor_ablation",
    "attention_visualization",
    "positional_encoding",
    "label_smoothing",
]

# All scalar metrics we want to pull per step
SCALAR_METRICS = [
    "train/loss",
    "train/lr",
    "val/loss",
    "val/bleu",
    "val/accuracy",
    "grad_norm/query",
    "grad_norm/key",
    "grad_norm/query_weight",
    "grad_norm/key_weight",
    "train/perplexity",
    "val/perplexity",
    "prediction_confidence",
    "train/grad_norm",
    "epoch",
    "step",
]

# Summary keys we want (best values, final values)
SUMMARY_KEYS = [
    "best_val_bleu",
    "best_val_loss",
    "final_val_bleu",
    "final_train_loss",
    "final_val_loss",
    "test_bleu",
]


def fetch_history(run, keys):
    """Fetch full history for a run, only columns that exist."""
    try:
        df = run.history(samples=10000, keys=keys, pandas=True)
        # Drop columns that are entirely NaN
        df = df.dropna(axis=1, how="all")
        return df.to_dict(orient="list")
    except Exception as e:
        print(f"  WARNING: Could not fetch full history for {run.name}: {e}")
        return {}


def fetch_files(run):
    """List files attached to the run (attention maps, etc.)."""
    try:
        files = [f.name for f in run.files()]
        return files
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="da6401-assignment-3")
    parser.add_argument("--entity", default=None, help="Your W&B entity/username")
    parser.add_argument("--output", default="wandb_runs_data.json")
    args = parser.parse_args()

    api = wandb.Api()
    path = f"{args.entity}/{args.project}" if args.entity else args.project

    print(f"Fetching runs from: {path}")
    try:
        runs = api.runs(path)
    except Exception as e:
        print(f"ERROR: Could not fetch runs. Check --entity and --project.\n{e}")
        return

    all_data = {}

    for run in runs:
        group = run.config.get("group", run.group or "ungrouped")
        run_name = run.config.get("run_name", run.name)

        print(f"  Fetching: [{group}] {run_name}  (id={run.id}, state={run.state})")

        # Full scalar history
        history = fetch_history(run, SCALAR_METRICS)

        # Summary / best values
        summary = {}
        for k in SUMMARY_KEYS:
            if k in run.summary:
                summary[k] = run.summary[k]
        # Also grab everything in summary just in case keys differ
        for k, v in run.summary.items():
            if not k.startswith("_") and isinstance(v, (int, float, str)):
                summary[k] = v

        # Config snapshot
        config_snapshot = {
            k: v for k, v in run.config.items()
            if not k.startswith("_") and isinstance(v, (int, float, str, bool))
        }

        # Files list
        files = fetch_files(run)

        entry = {
            "run_id": run.id,
            "run_name": run_name,
            "group": group,
            "state": run.state,
            "url": run.url,
            "config": config_snapshot,
            "summary": summary,
            "history": history,
            "files": files,
        }

        if group not in all_data:
            all_data[group] = {}
        all_data[group][run_name] = entry

    # Write output
    with open(args.output, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"\nDone. Saved to: {args.output}")
    print(f"Groups found: {list(all_data.keys())}")
    total_runs = sum(len(v) for v in all_data.values())
    print(f"Total runs fetched: {total_runs}")

    # Quick sanity check
    missing = [g for g in EXPECTED_GROUPS if g not in all_data]
    if missing:
        print(f"\nWARNING: Missing expected groups: {missing}")
    else:
        print("All expected experiment groups are present.")


if __name__ == "__main__":
    main()