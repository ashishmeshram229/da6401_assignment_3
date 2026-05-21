import argparse
import os
from copy import deepcopy

import torch

from train import DEFAULT_CONFIG, run_training


def base_colab_config(args):
    config = deepcopy(DEFAULT_CONFIG)
    config.update(
        {
            "wandb_project": args.project,
            "wandb_entity": args.entity,
            "use_wandb": True,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.accumulation,
            "mixed_precision": True,
            "num_workers": args.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "keep_dataset_in_memory": True,
            "eval_every": 1,
            "log_every": 25,
            "sample_limit": 4,
            "bleu_max_samples": args.bleu_samples,
            "eval_max_batches": args.eval_batches,
            "decode_max_len": 80,
            "log_qk_grad_steps": 1000,
            "attention_rollout_logging": True,
        }
    )
    return config


def report_experiments(base):
    experiments = []

    def add(group, name, **updates):
        config = deepcopy(base)
        config.update({"group": group, "run_name": name})
        config.update(updates)
        experiments.append(config)

    add("noam_vs_fixed_lr", "noam_scheduler", use_noam=True, learning_rate=1e-4)
    add("noam_vs_fixed_lr", "fixed_lr_1e_4", use_noam=False, learning_rate=1e-4)

    add("scaling_factor_ablation", "with_sqrt_dk_scaling", use_scaling=True)
    add("scaling_factor_ablation", "without_sqrt_dk_scaling", use_scaling=False)

    add("attention_visualization", "attention_head_analysis", use_noam=True, use_scaling=True)

    add("positional_encoding", "sinusoidal_position", positional_encoding="sinusoidal")
    add("positional_encoding", "learned_position", positional_encoding="learned")

    add("label_smoothing", "label_smoothing_0_1", label_smoothing=0.1)
    add("label_smoothing", "label_smoothing_0_0", label_smoothing=0.0)

    return experiments


def apply_profile(args):
    if args.profile == "fast":
        args.epochs = args.epochs or 4
        args.batch_size = args.batch_size or 64
        args.accumulation = args.accumulation or 1
        args.bleu_samples = args.bleu_samples or 100
        args.eval_batches = args.eval_batches or 20
    elif args.profile == "pro":
        args.epochs = args.epochs or 10
        args.batch_size = args.batch_size or 64
        args.accumulation = args.accumulation or 1
        args.bleu_samples = args.bleu_samples or 300
        args.eval_batches = args.eval_batches or None
    else:
        args.epochs = args.epochs or 25
        args.batch_size = args.batch_size or 64
        args.accumulation = args.accumulation or 1
        args.bleu_samples = args.bleu_samples or None
        args.eval_batches = args.eval_batches or None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="da6401-assignment-3")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--profile", choices=["fast", "pro", "full"], default="pro")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accumulation", type=int, default=None)
    parser.add_argument("--bleu-samples", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--only", default="all")
    return parser.parse_args()


def main():
    args = parse_args()
    apply_profile(args)

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. In Colab, set Runtime > Change runtime type > T4 GPU.")
    else:
        print("Using CUDA device:", torch.cuda.get_device_name(0))

    if not os.environ.get("WANDB_API_KEY"):
        print("WARNING: WANDB_API_KEY is not set. Run `import os; os.environ['WANDB_API_KEY']='...'` first.")

    base = base_colab_config(args)
    experiments = report_experiments(base)

    if args.only != "all":
        wanted = {item.strip() for item in args.only.split(",")}
        experiments = [
            config for config in experiments
            if config["group"] in wanted or config["run_name"] in wanted
        ]

    for idx, config in enumerate(experiments, start=1):
        print(f"\n[{idx}/{len(experiments)}] {config['group']} / {config['run_name']}")
        run_training(config)


if __name__ == "__main__":
    main()
