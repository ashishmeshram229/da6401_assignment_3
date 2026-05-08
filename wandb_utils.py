import os
from typing import Dict, List, Optional

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "matplotlib_cache"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import torch


def init_wandb(config: Dict, run_name: Optional[str] = None, group: Optional[str] = None):
    if config.get("use_wandb", True) is False:
        return None

    import wandb

    try:
        return wandb.init(
            project=config.get("wandb_project", "da6401-assignment-3"),
            entity=config.get("wandb_entity"),
            name=run_name or config.get("run_name"),
            group=group or config.get("group"),
            config=config,
        )
    except Exception as exc:
        print(f"W&B could not start, so training will continue without online logging: {exc}")
        return None


def log_metrics(run, metrics: Dict, step: Optional[int] = None) -> None:
    if run is None:
        return

    run.log(metrics, step=step)


def log_translation_table(run, rows: List[List[str]], step: int, name: str = "sample_translations") -> None:
    if run is None or not rows:
        return

    import wandb

    table = wandb.Table(columns=["German", "Reference", "Prediction"])
    for row in rows:
        table.add_data(row[0], row[1], row[2])

    run.log({name: table}, step=step)


def attention_entropy(attention: torch.Tensor) -> float:
    attention = attention.detach().float().clamp_min(1e-9)
    entropy = -(attention * attention.log()).sum(dim=-1)
    return entropy.mean().item()


def head_specialization_stats(attention_maps: List[torch.Tensor], prefix: str) -> Dict[str, float]:
    stats = {}

    for layer_idx, attention in enumerate(attention_maps):
        if attention.dim() != 4:
            continue

        attention = attention.detach().float()
        for head_idx in range(attention.size(1)):
            head = attention[:, head_idx]
            stats[f"{prefix}/layer_{layer_idx}_head_{head_idx}_entropy"] = attention_entropy(head)

            if head.size(-1) == head.size(-2):
                diag = torch.diagonal(head, dim1=-2, dim2=-1).mean().item()
                stats[f"{prefix}/layer_{layer_idx}_head_{head_idx}_diagonal_focus"] = diag

    return stats


def make_attention_figure(
    attention: torch.Tensor,
    source_tokens: List[str],
    target_tokens: List[str],
    title: str,
):
    attention = attention.detach().cpu().float()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attention.numpy(), aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(len(source_tokens)))
    ax.set_yticks(range(len(target_tokens)))
    ax.set_xticklabels(source_tokens, rotation=45, ha="right")
    ax.set_yticklabels(target_tokens)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def log_attention_heatmap(
    run,
    attention_maps: Dict,
    source_tokens: List[str],
    target_tokens: List[str],
    step: int,
    layer: int = -1,
    head: int = 0,
    name: str = "attention/cross_heatmap",
) -> None:
    if run is None:
        return

    cross_maps = attention_maps.get("decoder_cross", [])
    if not cross_maps:
        return

    import wandb

    attention = cross_maps[layer][0, head]
    attention = attention[: len(target_tokens), : len(source_tokens)]

    fig = make_attention_figure(
        attention,
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        title=f"Layer {layer}, Head {head}",
    )
    run.log({name: wandb.Image(fig)}, step=step)
    plt.close(fig)


def save_attention_heatmap(
    path: str,
    attention: torch.Tensor,
    source_tokens: List[str],
    target_tokens: List[str],
    title: str = "Attention",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = make_attention_figure(attention, source_tokens, target_tokens, title)
    fig.savefig(path, dpi=160)
    plt.close(fig)
