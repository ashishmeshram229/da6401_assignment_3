import math
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def shift_target(tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return tgt[:, :-1], tgt[:, 1:]


def gradient_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.detach().data.norm(2).item()
        total += param_norm * param_norm
    return math.sqrt(total)


def prediction_confidence(logits: torch.Tensor, target: torch.Tensor, pad_idx: int) -> float:
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        mask = target != pad_idx
        if mask.sum().item() == 0:
            return 0.0

        target_probs = probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        return target_probs[mask].mean().item()


def token_accuracy(logits: torch.Tensor, target: torch.Tensor, pad_idx: int) -> float:
    with torch.no_grad():
        predictions = logits.argmax(dim=-1)
        mask = target != pad_idx
        if mask.sum().item() == 0:
            return 0.0
        correct = (predictions == target) & mask
        return correct.sum().item() / mask.sum().item()


def named_gradient_norm(model: torch.nn.Module, include_terms: Tuple[str, ...]) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not any(term in name for term in include_terms):
            continue
        param_norm = parameter.grad.detach().data.norm(2).item()
        total += param_norm * param_norm
    return math.sqrt(total)


def strip_special_tokens(ids: List[int], vocab) -> List[int]:
    skipped = {vocab.pad_idx, vocab.sos_idx, vocab.eos_idx}
    return [idx for idx in ids if idx not in skipped]


def ngram_counts(tokens: List[int], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(len(tokens) - n + 1, 0)))


def sentence_bleu(reference: List[int], hypothesis: List[int], max_n: int = 4) -> float:
    if len(hypothesis) == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        hyp_counts = ngram_counts(hypothesis, n)
        ref_counts = ngram_counts(reference, n)

        if not hyp_counts:
            precisions.append(0.0)
            continue

        overlap = 0
        for ngram, count in hyp_counts.items():
            overlap += min(count, ref_counts.get(ngram, 0))

        precisions.append((overlap + 1.0) / (sum(hyp_counts.values()) + 1.0))

    log_precision = sum(math.log(p) for p in precisions) / max_n
    brevity = 1.0
    if len(hypothesis) < len(reference):
        brevity = math.exp(1.0 - len(reference) / max(len(hypothesis), 1))

    return 100.0 * brevity * math.exp(log_precision)


def corpus_bleu(references: List[List[int]], hypotheses: List[List[int]], max_n: int = 4) -> float:
    if not references or not hypotheses:
        return 0.0

    total_score = 0.0
    for reference, hypothesis in zip(references, hypotheses):
        total_score += sentence_bleu(reference, hypothesis, max_n=max_n)

    return total_score / len(hypotheses)


@torch.no_grad()
def evaluate_loss(model, data_loader, criterion, device, pad_idx: int, max_batches: Optional[int] = None) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0

    for batch_idx, (src, tgt) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        src = src.to(device)
        tgt = tgt.to(device)
        decoder_input, expected = shift_target(tgt)

        logits, _ = model(src, decoder_input)
        if criterion.__class__.__name__ == "LabelSmoothingLoss":
            loss = criterion(logits, expected)
        else:
            vocab_size = logits.size(-1)
            loss = criterion(logits.reshape(-1, vocab_size), expected.reshape(-1))

        total_loss += loss.item()
        batches += 1

    return total_loss / max(batches, 1)


@torch.no_grad()
def evaluate_token_metrics(
    model,
    data_loader,
    device,
    pad_idx: int,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    model.eval()
    total_accuracy = 0.0
    total_confidence = 0.0
    batches = 0

    for batch_idx, (src, tgt) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        src = src.to(device)
        tgt = tgt.to(device)
        decoder_input, expected = shift_target(tgt)
        logits, _ = model(src, decoder_input)

        total_accuracy += token_accuracy(logits, expected, pad_idx)
        total_confidence += prediction_confidence(logits, expected, pad_idx)
        batches += 1

    return total_accuracy / max(batches, 1), total_confidence / max(batches, 1)


@torch.no_grad()
def evaluate_bleu(
    model,
    data_loader,
    tgt_vocab,
    device,
    max_len: int = 80,
    max_batches: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> float:
    model.eval()
    references = []
    hypotheses = []

    for batch_idx, (src, tgt) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        src = src.to(device)
        for row in range(src.size(0)):
            if max_samples is not None and len(hypotheses) >= max_samples:
                return corpus_bleu(references, hypotheses)

            source_ids = src[row]
            source_ids = source_ids[source_ids != model.src_pad_idx].unsqueeze(0)
            output_ids, _ = model.greedy_decode(source_ids, max_len=max_len)

            ref_ids = strip_special_tokens(tgt[row].tolist(), tgt_vocab)
            hyp_ids = strip_special_tokens(output_ids, tgt_vocab)

            references.append(ref_ids)
            hypotheses.append(hyp_ids)

    return corpus_bleu(references, hypotheses)


@torch.no_grad()
def sample_translations(model, data_loader, src_vocab, tgt_vocab, device, limit: int = 5, max_len: int = 80):
    model.eval()
    rows = []

    for src, tgt in data_loader:
        src = src.to(device)
        for i in range(src.size(0)):
            if len(rows) >= limit:
                return rows

            source_ids = src[i]
            source_ids = source_ids[source_ids != model.src_pad_idx].unsqueeze(0)
            output_ids, _ = model.greedy_decode(source_ids, max_len=max_len)

            german = src_vocab.decode(source_ids.squeeze(0).cpu().tolist())
            reference = tgt_vocab.decode(tgt[i].tolist())
            prediction = tgt_vocab.decode(output_ids)
            rows.append([german, reference, prediction])

    return rows


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_bleu: float,
    config: Dict,
    src_vocab=None,
    tgt_vocab=None,
) -> None:
    src_vocab_data = None
    tgt_vocab_data = None
    if src_vocab is not None:
        src_vocab_data = {"stoi": src_vocab.stoi, "itos": src_vocab.itos}
    if tgt_vocab is not None:
        tgt_vocab_data = {"stoi": tgt_vocab.stoi, "itos": tgt_vocab.itos}

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_bleu": best_bleu,
            "config": config,
            "src_vocab": src_vocab_data,
            "tgt_vocab": tgt_vocab_data,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        path,
    )
