import argparse
import os
from collections import defaultdict
from datetime import datetime

import wandb
from wandb.apis.reports import v2


DEFAULT_PROJECT = "da6401-assignment-3"


def login_if_needed():
    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)


def get_entity(entity_arg):
    if entity_arg:
        return entity_arg

    api = wandb.Api()
    if getattr(api, "default_entity", None):
        return api.default_entity

    viewer = api.viewer()
    return viewer.get("entity") or viewer.get("username")


def collect_run_summary(entity, project):
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}"))
    groups = defaultdict(list)

    for run in runs:
        groups[run.group or "ungrouped"].append(run)

    lines = []
    lines.append("| Group | Run | State | Best/Final BLEU | Final loss | Notes |")
    lines.append("|---|---:|---|---:|---:|---|")

    for group in sorted(groups):
        for run in sorted(groups[group], key=lambda r: r.created_at or ""):
            summary = run.summary
            bleu = summary.get("valid/bleu", summary.get("test/bleu", ""))
            loss = summary.get("train/epoch_loss", summary.get("train/loss", ""))
            notes = []
            if "use_noam" in run.config:
                notes.append(f"noam={run.config.get('use_noam')}")
            if "use_scaling" in run.config:
                notes.append(f"scale={run.config.get('use_scaling')}")
            if "positional_encoding" in run.config:
                notes.append(f"pos={run.config.get('positional_encoding')}")
            if "label_smoothing" in run.config:
                notes.append(f"ls={run.config.get('label_smoothing')}")

            lines.append(
                f"| {group} | {run.name} | {run.state} | {format_value(bleu)} | "
                f"{format_value(loss)} | {', '.join(notes)} |"
            )

    return "\n".join(lines), runs


def format_value(value):
    if value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def runset(entity, project, name, group):
    return v2.Runset(
        entity=entity,
        project=project,
        name=name,
        query=f'group:"{group}"',
    )


def line(title, y, layout, groupby=None, smoothing=0.4):
    return v2.LinePlot(
        title=title,
        x="Step",
        y=y,
        groupby=groupby,
        smoothing_factor=smoothing,
        smoothing_type="exponential",
        legend_template="${run:name}",
        layout=layout,
    )


def scalar(title, metric, layout):
    return v2.ScalarChart(
        title=title,
        metric=metric,
        groupby_aggfunc="max",
        layout=layout,
    )


def build_report(entity, project, title, run_table):
    noam = runset(entity, project, "Noam Scheduler vs Fixed LR", "noam_vs_fixed_lr")
    scaling = runset(entity, project, "Scaling Factor Ablation", "scaling_factor_ablation")
    attention = runset(entity, project, "Attention Head Visualization", "attention_visualization")
    position = runset(entity, project, "Sinusoidal vs Learned Position", "positional_encoding")
    smoothing = runset(entity, project, "Label Smoothing", "label_smoothing")

    blocks = [
        v2.H1(title),
        v2.P(
            "DA6401 Assignment 3: German to English machine translation with a Transformer "
            "implemented from scratch in PyTorch. This report focuses on optimization stability, "
            "attention behavior, positional information, and decoder calibration."
        ),
        v2.TableOfContents(),
        v2.H2("Experiment Inventory"),
        v2.MarkdownBlock(
            "The table below is generated directly from W&B run metadata and summaries.\n\n"
            + run_table
        ),
        v2.H2("1. The Necessity of the Noam Scheduler"),
        v2.MarkdownBlock(
            "The Transformer is sensitive to the initial learning rate because early self-attention "
            "weights are poorly calibrated. Large updates in the first few hundred steps can push "
            "query-key dot products into saturated softmax regions and destabilize residual streams. "
            "The Noam schedule avoids this by linearly warming up the learning rate before switching "
            "to inverse-square-root decay. The fixed learning-rate baseline tests whether this warmup "
            "is merely convenient or actually stabilizing."
        ),
        v2.PanelGrid(
            runsets=[noam],
            panels=[
                line("Training loss: Noam vs fixed LR", ["train/loss"], v2.Layout(w=12, h=6)),
                line("Validation token accuracy: Noam vs fixed LR", ["valid/token_accuracy"], v2.Layout(w=12, h=6)),
                line("Validation BLEU: Noam vs fixed LR", ["valid/bleu"], v2.Layout(w=12, h=6), smoothing=0.0),
                line("Learning-rate trajectory", ["train/learning_rate"], v2.Layout(w=12, h=6), smoothing=0.0),
            ],
        ),
        v2.H2("2. Ablation: Scaling Factor 1/sqrt(dk)"),
        v2.MarkdownBlock(
            "Without the scaling term, dot products grow with the key dimension and can push the "
            "softmax into low-gradient regions. The first 1,000 optimizer steps are the most useful "
            "region to inspect because the model has not yet learned stable query/key projections. "
            "We log global gradient norms plus separate Query and Key projection gradient norms."
        ),
        v2.PanelGrid(
            runsets=[scaling],
            panels=[
                line("Training loss with and without scaling", ["train/loss"], v2.Layout(w=12, h=6)),
                line("Query gradient norm, first 1k steps", ["gradients/query_norm"], v2.Layout(w=12, h=6), smoothing=0.2),
                line("Key gradient norm, first 1k steps", ["gradients/key_norm"], v2.Layout(w=12, h=6), smoothing=0.2),
                line("Global gradient norm", ["train/gradient_norm"], v2.Layout(w=12, h=6), smoothing=0.2),
            ],
        ),
        v2.H2("3. Attention Rollout and Head Specialization"),
        v2.MarkdownBlock(
            "The final encoder layer is visualized head by head. Heads with strong diagonal mass "
            "usually behave like local alignment or identity-preserving heads; heads with broad or "
            "off-diagonal mass often capture longer-range dependencies. Entropy and diagonal-focus "
            "metrics help identify redundancy: multiple heads with similar entropy and diagonal "
            "patterns may be learning overlapping roles."
        ),
        v2.PanelGrid(
            runsets=[attention],
            panels=[
                v2.MediaBrowser(
                    title="Last encoder layer: individual head heatmaps",
                    media_keys=[f"attention/encoder_last_head_{i}" for i in range(8)],
                    num_columns=4,
                    layout=v2.Layout(w=12, h=8),
                ),
                v2.MediaBrowser(
                    title="Decoder cross-attention heatmap",
                    media_keys=["attention/cross_heatmap"],
                    num_columns=2,
                    layout=v2.Layout(w=12, h=6),
                ),
                line(
                    "Encoder head entropy",
                    [f"attention/encoder/layer_3_head_{i}_entropy" for i in range(8)],
                    v2.Layout(w=12, h=6),
                    smoothing=0.0,
                ),
                line(
                    "Encoder head diagonal focus",
                    [f"attention/encoder/layer_3_head_{i}_diagonal_focus" for i in range(8)],
                    v2.Layout(w=12, h=6),
                    smoothing=0.0,
                ),
            ],
        ),
        v2.H2("4. Sinusoidal vs Learned Positional Encoding"),
        v2.MarkdownBlock(
            "Learned positional embeddings can specialize to the training length distribution, while "
            "sinusoidal encodings provide a deterministic basis where relative offsets correspond to "
            "linear transformations of positions. This is why sinusoidal encodings can theoretically "
            "extrapolate to longer sequences than those seen during training, provided the model has "
            "learned to use the frequency structure."
        ),
        v2.PanelGrid(
            runsets=[position],
            panels=[
                line("Validation BLEU by positional encoding", ["valid/bleu"], v2.Layout(w=12, h=6), smoothing=0.0),
                line("Validation token accuracy", ["valid/token_accuracy"], v2.Layout(w=12, h=6), smoothing=0.0),
                scalar("Best validation BLEU", "valid/bleu", v2.Layout(w=6, h=4)),
                scalar("Best validation token accuracy", "valid/token_accuracy", v2.Layout(w=6, h=4)),
            ],
        ),
        v2.H2("5. Decoder Sensitivity: Label Smoothing"),
        v2.MarkdownBlock(
            "Label smoothing replaces the one-hot target with a softened distribution. This prevents "
            "the decoder from assigning nearly all probability mass to a single token too early. The "
            "result is usually lower confidence and better calibration, even when the smoothed loss or "
            "perplexity appears worse than plain cross-entropy."
        ),
        v2.PanelGrid(
            runsets=[smoothing],
            panels=[
                line("Correct-token prediction confidence", ["train/correct_token_confidence"], v2.Layout(w=12, h=6)),
                line("Validation correct-token confidence", ["valid/correct_token_confidence"], v2.Layout(w=12, h=6)),
                line("Validation BLEU", ["valid/bleu"], v2.Layout(w=12, h=6), smoothing=0.0),
                line("Training loss", ["train/loss"], v2.Layout(w=12, h=6)),
            ],
        ),
        v2.H2("Conclusion"),
        v2.MarkdownBlock(
            "The strongest runs should combine Noam scheduling, scaled dot-product attention, "
            "sinusoidal positional encoding or the empirically stronger positional variant, and "
            "label smoothing. The report panels are intentionally interactive so that individual "
            "runs, heads, and training phases can be inspected rather than summarized only by final BLEU."
        ),
    ]

    return v2.Report(
        entity=entity,
        project=project,
        title=title,
        description="DA6401 Assignment 3 Transformer MT experiment report.",
        width="fluid",
        blocks=blocks,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--title", default=None)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    login_if_needed()
    entity = get_entity(args.entity)
    title = args.title or f"DA6401 Assignment 3 Transformer Report ({datetime.now().strftime('%Y-%m-%d')})"
    run_table, runs = collect_run_summary(entity, args.project)

    if not runs:
        raise RuntimeError(
            f"No runs found in {entity}/{args.project}. Run the experiment suite before creating the report."
        )

    report = build_report(entity, args.project, title, run_table)
    report.save(draft=args.draft)

    print("report_url:", report.url)
    if args.share:
        print("share_url:", report.enable_share_link())


if __name__ == "__main__":
    main()
