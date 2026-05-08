import argparse
import os
import time
from copy import deepcopy

import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import make_dataloaders
from model import LabelSmoothingLoss, Transformer
from scheduler import make_scheduler
from utils import (
    evaluate_bleu,
    evaluate_loss,
    get_device,
    gradient_norm,
    prediction_confidence,
    sample_translations,
    save_checkpoint,
    set_seed,
    shift_target,
)
from wandb_utils import (
    head_specialization_stats,
    init_wandb,
    log_attention_heatmap,
    log_metrics,
    log_translation_table,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


DEFAULT_CONFIG = {
    "seed": 42,
    "epochs": 25,
    "batch_size": 32,
    "gradient_accumulation_steps": 2,
    "mixed_precision": False,
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 4,
    "d_ff": 1024,
    "dropout": 0.1,
    "max_len": 100,
    "decode_max_len": 80,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "clip_grad": 1.0,
    "use_noam": True,
    "warmup_steps": 4000,
    "noam_factor": 1.0,
    "use_scaling": True,
    "label_smoothing": 0.1,
    "positional_encoding": "sinusoidal",
    "min_freq": 2,
    "max_vocab_size": 12000,
    "num_workers": 0,
    "eval_max_batches": None,
    "bleu_max_batches": None,
    "bleu_max_samples": None,
    "train_max_batches": None,
    "log_every": 50,
    "eval_every": 1,
    "sample_limit": 5,
    "attention_rollout_logging": True,
    "use_wandb": True,
    "wandb_project": "da6401-assignment-3",
    "group": "main_training",
    "run_name": "transformer_noam_scaling_sinusoidal",
}


def ids_to_tokens(ids, vocab):
    tokens = []
    for idx in ids:
        idx = int(idx)
        if idx == vocab.pad_idx:
            continue
        if 0 <= idx < len(vocab.itos):
            tokens.append(vocab.itos[idx])
    return tokens


def make_model(config, src_vocab, tgt_vocab, device):
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        max_len=config["max_len"],
        src_pad_idx=src_vocab.pad_idx,
        tgt_pad_idx=tgt_vocab.pad_idx,
        positional_encoding=config["positional_encoding"],
        use_scaling=config["use_scaling"],
        src_vocab_path=os.path.join(BASE_DIR, "vocab", "src_vocab.json"),
        tgt_vocab_path=os.path.join(BASE_DIR, "vocab", "tgt_vocab.json"),
        weight_path=os.path.join(BASE_DIR, "checkpoints", "transformer_best.pt"),
        load_weights=False,
        device=device,
    )
    return model


def make_loss(config, pad_idx):
    smoothing = config.get("label_smoothing", 0.0)
    if smoothing > 0:
        return LabelSmoothingLoss(smoothing=smoothing, ignore_index=pad_idx)
    return nn.CrossEntropyLoss(ignore_index=pad_idx)


def compute_loss(criterion, logits, expected):
    if isinstance(criterion, LabelSmoothingLoss):
        return criterion(logits, expected)

    vocab_size = logits.size(-1)
    return criterion(logits.reshape(-1, vocab_size), expected.reshape(-1))


def log_attention_examples(run, model, valid_loader, src_vocab, tgt_vocab, device, step, config):
    if run is None:
        return

    model.eval()
    src, tgt = next(iter(valid_loader))
    src = src[:1].to(device)
    tgt = tgt[:1].to(device)
    decoder_input, _ = shift_target(tgt)

    with torch.no_grad():
        _, attention_maps = model(src, decoder_input)

    source_tokens = ids_to_tokens(src[0].cpu().tolist(), src_vocab)
    target_tokens = ids_to_tokens(decoder_input[0].cpu().tolist(), tgt_vocab)

    log_attention_heatmap(
        run,
        attention_maps,
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        step=step,
        layer=-1,
        head=0,
    )

    stats = {}
    stats.update(head_specialization_stats(attention_maps["encoder"], "attention/encoder"))
    stats.update(head_specialization_stats(attention_maps["decoder_cross"], "attention/cross"))
    log_metrics(run, stats, step=step)

    if config.get("attention_rollout_logging", True):
        rollout = model.get_attention_rollout(attention_maps["encoder"])
        rollout_score = rollout[0].mean().item()
        log_metrics(run, {"attention/encoder_rollout_mean": rollout_score}, step=step)


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    config,
    run,
    epoch,
    global_step,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_confidence = 0.0
    batches = 0
    accumulation_steps = config.get("gradient_accumulation_steps", 1)

    progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for batch_idx, (src, tgt) in enumerate(progress):
        if config.get("train_max_batches") is not None and batch_idx >= config["train_max_batches"]:
            break

        src = src.to(device)
        tgt = tgt.to(device)

        decoder_input, expected = shift_target(tgt)
        logits, _ = model(src, decoder_input)
        loss = compute_loss(criterion, logits, expected)
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        confidence = prediction_confidence(logits.detach(), expected, model.tgt_pad_idx)
        total_loss += loss.item()
        total_confidence += confidence
        batches += 1

        should_step = (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader)
        if should_step:
            grad_norm = gradient_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
            optimizer.step()
            lr = scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % config["log_every"] == 0:
                log_metrics(
                    run,
                    {
                        "train/loss": total_loss / batches,
                        "train/prediction_confidence": total_confidence / batches,
                        "train/gradient_norm": grad_norm,
                        "train/learning_rate": lr,
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        progress.set_postfix(loss=total_loss / batches)

    return total_loss / max(batches, 1), total_confidence / max(batches, 1), global_step


def run_training(config):
    config = deepcopy(config)
    set_seed(config["seed"])
    device = get_device()

    vocab_dir = os.path.join(BASE_DIR, "vocab")
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")

    train_loader, valid_loader, test_loader, src_vocab, tgt_vocab = make_dataloaders(
        batch_size=config["batch_size"],
        min_freq=config["min_freq"],
        max_vocab_size=config["max_vocab_size"],
        max_len=config["max_len"],
        num_workers=config["num_workers"],
        vocab_dir=vocab_dir,
    )

    model = make_model(config, src_vocab, tgt_vocab, device)
    criterion = make_loss(config, tgt_vocab.pad_idx)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config["weight_decay"],
    )
    scheduler = make_scheduler(optimizer, config)

    run = init_wandb(config, run_name=config.get("run_name"), group=config.get("group"))

    best_bleu = -1.0
    global_step = 0
    start_time = time.time()
    best_path = os.path.join(checkpoint_dir, "transformer_best.pt")
    last_path = os.path.join(checkpoint_dir, "transformer_last.pt")

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_confidence, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            run=run,
            epoch=epoch,
            global_step=global_step,
        )

        if epoch % config["eval_every"] == 0:
            valid_loss = evaluate_loss(
                model,
                valid_loader,
                criterion,
                device,
                pad_idx=tgt_vocab.pad_idx,
                max_batches=config["eval_max_batches"],
            )
            valid_bleu = evaluate_bleu(
                model,
                valid_loader,
                tgt_vocab,
                device,
                max_len=config["decode_max_len"],
                max_batches=config["bleu_max_batches"],
                max_samples=config["bleu_max_samples"],
            )
            rows = sample_translations(
                model,
                valid_loader,
                src_vocab,
                tgt_vocab,
                device,
                limit=config["sample_limit"],
                max_len=config["decode_max_len"],
            )

            metrics = {
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "train/epoch_prediction_confidence": train_confidence,
                "valid/loss": valid_loss,
                "valid/bleu": valid_bleu,
                "time/minutes": (time.time() - start_time) / 60.0,
            }
            log_metrics(run, metrics, step=global_step)
            log_translation_table(run, rows, step=global_step)
            log_attention_examples(run, model, valid_loader, src_vocab, tgt_vocab, device, global_step, config)

            print(
                f"Epoch {epoch:02d} | "
                f"train loss {train_loss:.4f} | "
                f"valid loss {valid_loss:.4f} | "
                f"valid BLEU {valid_bleu:.2f}"
            )

            if valid_bleu > best_bleu:
                best_bleu = valid_bleu
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_bleu,
                    config,
                    src_vocab=src_vocab,
                    tgt_vocab=tgt_vocab,
                )
                print(f"Saved best checkpoint with BLEU {best_bleu:.2f}")

        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_bleu,
            config,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
        )

    if test_loader is not None:
        test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device, max_len=config["decode_max_len"])
        log_metrics(run, {"test/bleu": test_bleu}, step=global_step)
        print(f"Test BLEU: {test_bleu:.2f}")

    if run is not None:
        run.finish()

    return best_bleu


def experiment_configs(base_config):
    experiments = []

    noam_config = deepcopy(base_config)
    noam_config.update(
        {
            "group": "noam_vs_fixed_lr",
            "run_name": "noam_scheduler",
            "use_noam": True,
        }
    )
    experiments.append(noam_config)

    fixed_config = deepcopy(base_config)
    fixed_config.update(
        {
            "group": "noam_vs_fixed_lr",
            "run_name": "fixed_lr",
            "use_noam": False,
            "learning_rate": 1e-4,
        }
    )
    experiments.append(fixed_config)

    scaled_config = deepcopy(base_config)
    scaled_config.update(
        {
            "group": "scaling_factor_ablation",
            "run_name": "with_sqrt_dk_scaling",
            "use_scaling": True,
        }
    )
    experiments.append(scaled_config)

    no_scale_config = deepcopy(base_config)
    no_scale_config.update(
        {
            "group": "scaling_factor_ablation",
            "run_name": "without_sqrt_dk_scaling",
            "use_scaling": False,
        }
    )
    experiments.append(no_scale_config)

    sinusoidal_config = deepcopy(base_config)
    sinusoidal_config.update(
        {
            "group": "positional_encoding",
            "run_name": "sinusoidal_position",
            "positional_encoding": "sinusoidal",
        }
    )
    experiments.append(sinusoidal_config)

    learned_config = deepcopy(base_config)
    learned_config.update(
        {
            "group": "positional_encoding",
            "run_name": "learned_position",
            "positional_encoding": "learned",
        }
    )
    experiments.append(learned_config)

    smoothing_config = deepcopy(base_config)
    smoothing_config.update(
        {
            "group": "label_smoothing",
            "run_name": "label_smoothing_0_1",
            "label_smoothing": 0.1,
        }
    )
    experiments.append(smoothing_config)

    no_smoothing_config = deepcopy(base_config)
    no_smoothing_config.update(
        {
            "group": "label_smoothing",
            "run_name": "label_smoothing_0_0",
            "label_smoothing": 0.0,
        }
    )
    experiments.append(no_smoothing_config)

    return experiments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--accumulation", type=int, default=DEFAULT_CONFIG["gradient_accumulation_steps"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--experiments", action="store_true")
    parser.add_argument("--run-name", type=str, default=DEFAULT_CONFIG["run_name"])
    parser.add_argument("--group", type=str, default=DEFAULT_CONFIG["group"])
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_CONFIG["wandb_project"])
    return parser.parse_args()


def main():
    args = parse_args()
    config = deepcopy(DEFAULT_CONFIG)

    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["gradient_accumulation_steps"] = args.accumulation
    config["learning_rate"] = args.lr
    config["use_wandb"] = not args.no_wandb
    config["run_name"] = args.run_name
    config["group"] = args.group
    config["wandb_project"] = args.wandb_project

    if args.quick:
        config["epochs"] = 1
        config["train_max_batches"] = 5
        config["eval_max_batches"] = 2
        config["bleu_max_batches"] = 1
        config["bleu_max_samples"] = 1
        config["sample_limit"] = 1
        config["decode_max_len"] = 20
        config["log_every"] = 5
        config["use_wandb"] = False

    if args.experiments:
        for exp_config in experiment_configs(config):
            print(f"Starting experiment: {exp_config['group']} / {exp_config['run_name']}")
            run_training(exp_config)
        return

    run_training(config)


if __name__ == "__main__":
    main()
