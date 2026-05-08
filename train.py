"""
train.py - Training pipeline for the Transformer NMT model
DA6401 Assignment 3

Runs all W&B experiments:
  1. Noam vs Fixed LR
  2. Scaling factor ablation
  3. Attention rollout / head visualisation
  4. Sinusoidal PE vs Learned Embeddings
  5. Label smoothing ablation
"""

import os
import math
import argparse
import torch
import torch.nn as nn
import numpy as np
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import get_dataloaders, PAD_IDX, SOS_IDX, EOS_IDX
from model import (Transformer, NoamScheduler, LabelSmoothingLoss,
                   make_padding_mask, make_causal_mask, PositionalEncoding)

try:
    from evaluate import load as eval_load
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False


# ─────────────────────────────────────────────
# BLEU evaluation
# ─────────────────────────────────────────────
def evaluate_bleu(model, loader, tgt_itos, device, max_samples=500):
    model.eval()
    predictions, references = [], []
    count = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask = make_padding_mask(src, PAD_IDX)
            enc_out  = model.encoder(src, src_mask)

            for i in range(src.size(0)):
                src_i    = src[i:i+1]
                mask_i   = src_mask[i:i+1]
                enc_i    = enc_out[i:i+1]

                pred_ids = greedy_decode_enc(model, src_i, mask_i, enc_i, device)
                pred_words = ids_to_words(pred_ids, tgt_itos)
                ref_ids    = tgt[i].tolist()
                ref_words  = ids_to_words(ref_ids, tgt_itos)

                predictions.append(pred_words)
                references.append([ref_words])
                count += 1
                if count >= max_samples:
                    break
            if count >= max_samples:
                break

    if BLEU_AVAILABLE:
        bleu_metric = eval_load("bleu")
        result = bleu_metric.compute(predictions=predictions, references=references)
        return result["bleu"] * 100
    else:
        return _simple_bleu(predictions, references)


def greedy_decode_enc(model, src, src_mask, enc_out, device, max_len=50):
    """Greedy decode given pre-computed encoder output."""
    tgt = torch.tensor([[SOS_IDX]], device=device)
    for _ in range(max_len):
        tgt_mask = make_causal_mask(tgt.size(1), device)
        dec_out  = model.decoder(tgt, enc_out, src_mask, tgt_mask)
        logits   = model.fc_out(dec_out[:, -1, :])
        next_tok = logits.argmax(dim=-1, keepdim=True)
        if next_tok.item() == EOS_IDX:
            break
        tgt = torch.cat([tgt, next_tok], dim=1)
    return tgt[0, 1:].tolist()


def ids_to_words(ids, itos):
    words = []
    for idx in ids:
        tok = itos[idx] if idx < len(itos) else "<unk>"
        if tok in ("<eos>", "<pad>"):
            break
        if tok not in ("<sos>",):
            words.append(tok)
    return words


def _simple_bleu(predictions, references):
    """Fallback corpus BLEU (1-gram precision) when evaluate not available."""
    from collections import Counter
    correct = total = 0
    for pred, refs in zip(predictions, references):
        ref = refs[0]
        ref_c = Counter(ref)
        pred_c = Counter(pred)
        for tok, cnt in pred_c.items():
            correct += min(cnt, ref_c.get(tok, 0))
        total += len(pred)
    return (correct / (total + 1e-9)) * 100


# ─────────────────────────────────────────────
# One epoch train / val
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scheduler, device, clip=1.0,
                log_grad_norms=False, step_offset=0):
    model.train()
    total_loss = 0
    n_batches  = 0
    grad_norms = []

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src, tgt_in)   # (batch, tgt_len-1, vocab)

        # Flatten for loss
        logits_flat  = logits.reshape(-1, logits.size(-1))
        targets_flat = tgt_out.reshape(-1)

        loss = criterion(logits_flat, targets_flat)
        loss.backward()

        if log_grad_norms:
            qk_norm = 0.0
            for layer in model.encoder.layers:
                for p in list(layer.self_attn.W_Q.parameters()) + \
                         list(layer.self_attn.W_K.parameters()):
                    if p.grad is not None:
                        qk_norm += p.grad.norm().item()
            grad_norms.append(qk_norm)

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches, grad_norms


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches  = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            logits  = model(src, tgt_in)
            logits_flat  = logits.reshape(-1, logits.size(-1))
            targets_flat = tgt_out.reshape(-1)
            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item()
            n_batches  += 1
    return total_loss / n_batches


# ─────────────────────────────────────────────
# Build model helper
# ─────────────────────────────────────────────
def build_model(src_vocab_size, tgt_vocab_size, cfg):
    """Instantiate a fresh Transformer using raw sizes (no weight loading)."""
    from model import Encoder, Decoder
    model = Transformer.__new__(Transformer)
    nn.Module.__init__(model)
    model.pad_idx       = PAD_IDX
    model.src_vocab     = {}
    model.tgt_vocab     = {}
    model.src_itos      = []
    model.tgt_itos      = []
    model.src_vocab_size = src_vocab_size
    model.tgt_vocab_size = tgt_vocab_size
    model.max_len       = cfg["max_len"]
    model.encoder = Encoder(src_vocab_size, cfg["d_model"], cfg["num_heads"],
                             cfg["num_layers"], cfg["d_ff"], cfg["max_len"], cfg["dropout"])
    model.decoder = Decoder(tgt_vocab_size, cfg["d_model"], cfg["num_heads"],
                             cfg["num_layers"], cfg["d_ff"], cfg["max_len"], cfg["dropout"])
    model.fc_out  = nn.Linear(cfg["d_model"], tgt_vocab_size)
    from model import Transformer as T
    T._init_weights(model)

    # Patch forward / greedy_decode
    import types
    model.forward = types.MethodType(Transformer.forward, model)
    model.greedy_decode = types.MethodType(Transformer.greedy_decode, model)
    return model


# ─────────────────────────────────────────────
# Experiment 1: Noam vs Fixed LR
# ─────────────────────────────────────────────
def exp_noam_vs_fixed(train_loader, val_loader, tgt_itos,
                      src_vocab_size, tgt_vocab_size, cfg, device, epochs=10):
    for use_noam in [True, False]:
        name = "noam" if use_noam else "fixed_lr"
        run = wandb.init(project=cfg["project"], name=f"exp1_{name}", reinit=True,
                         config={**cfg, "use_noam": use_noam})

        model = build_model(src_vocab_size, tgt_vocab_size, cfg).to(device)
        criterion = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX, smoothing=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

        scheduler = None
        if use_noam:
            scheduler = NoamScheduler(optimizer, cfg["d_model"], warmup_steps=cfg["warmup_steps"])
            # Initialise LR to step=1 value before first batch
            scheduler.step()

        for epoch in range(epochs):
            tr_loss, _ = train_epoch(model, train_loader, optimizer, criterion, scheduler, device)
            vl_loss    = val_epoch(model, val_loader, criterion, device)
            bleu       = evaluate_bleu(model, val_loader, tgt_itos, device, max_samples=200)
            lr_now     = optimizer.param_groups[0]["lr"]
            wandb.log({"train_loss": tr_loss, "val_loss": vl_loss,
                       "val_bleu": bleu, "lr": lr_now, "epoch": epoch + 1})
            print(f"[Exp1 {name}] Ep {epoch+1}/{epochs}  loss={tr_loss:.4f}  bleu={bleu:.2f}")

        run.finish()


# ─────────────────────────────────────────────
# Experiment 2: Scaling factor ablation
# ─────────────────────────────────────────────
def exp_scaling_ablation(train_loader, val_loader, src_vocab_size, tgt_vocab_size, cfg, device):
    """Train with / without 1/sqrt(dk) for the first 1000 steps, log grad norms."""
    from model import MultiHeadAttention, scaled_dot_product_attention
    import types

    for use_scale in [True, False]:
        name = "with_scale" if use_scale else "no_scale"
        run = wandb.init(project=cfg["project"], name=f"exp2_{name}", reinit=True,
                         config={**cfg, "use_scale": use_scale})

        model = build_model(src_vocab_size, tgt_vocab_size, cfg).to(device)

        # Monkey-patch attention if not using scale
        if not use_scale:
            from model import MultiHeadAttention as MHA
            original_fwd = MHA.forward

            def no_scale_fwd(self_mha, Q, K, V, mask=None):
                from model import F as _F
                import math
                Q2 = self_mha.split_heads(self_mha.W_Q(Q))
                K2 = self_mha.split_heads(self_mha.W_K(K))
                V2 = self_mha.split_heads(self_mha.W_V(V))
                # No sqrt(dk) scaling
                scores = torch.matmul(Q2, K2.transpose(-2, -1))
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, float('-inf'))
                attn_w = _F.softmax(scores, dim=-1)
                attn_w = torch.nan_to_num(attn_w, nan=0.0)
                out = torch.matmul(attn_w, V2)
                self_mha.attn_weights = attn_w
                b, _, s, _ = out.size()
                out = out.transpose(1, 2).contiguous().view(b, s, self_mha.d_model)
                return self_mha.W_O(out)

            for layer in model.encoder.layers:
                layer.self_attn.forward = types.MethodType(no_scale_fwd, layer.self_attn)

        criterion = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX, smoothing=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

        model.train()
        step = 0
        MAX_STEPS = 1000

        for src, tgt in train_loader:
            if step >= MAX_STEPS:
                break
            src, tgt = src.to(device), tgt.to(device)
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            optimizer.zero_grad()
            logits = model(src, tgt_in)
            loss   = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()

            # Log Q/K grad norms
            qk_norm = 0.0
            for layer in model.encoder.layers:
                for p in list(layer.self_attn.W_Q.parameters()) + \
                         list(layer.self_attn.W_K.parameters()):
                    if p.grad is not None:
                        qk_norm += p.grad.norm().item()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            wandb.log({"train_loss": loss.item(), "qk_grad_norm": qk_norm, "step": step})
            if step % 100 == 0:
                print(f"[Exp2 {name}] step {step}  loss={loss.item():.4f}  qk_norm={qk_norm:.4f}")

        run.finish()


# ─────────────────────────────────────────────
# Experiment 3: Attention rollout
# ─────────────────────────────────────────────
def exp_attention_rollout(model, val_loader, src_itos, tgt_itos, device, cfg):
    """Visualise per-head attention maps from last encoder layer."""
    run = wandb.init(project=cfg["project"], name="exp3_attention_rollout", reinit=True,
                     config=cfg)
    model.eval()

    src, tgt = next(iter(val_loader))
    src = src[:1].to(device)
    src_mask = make_padding_mask(src, PAD_IDX)

    with torch.no_grad():
        model.encoder(src, src_mask)

    last_layer = model.encoder.layers[-1]
    attn_weights = last_layer.self_attn.attn_weights  # (1, heads, seq, seq)
    attn_weights = attn_weights[0].cpu().numpy()       # (heads, seq, seq)

    src_tokens = [src_itos[i] if i < len(src_itos) else "<unk>"
                  for i in src[0].cpu().tolist()]

    num_heads = attn_weights.shape[0]
    fig, axes = plt.subplots(2, num_heads // 2, figsize=(4 * num_heads // 2, 8))
    axes = axes.flatten()

    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attn_weights[h], cmap="viridis", aspect="auto")
        ax.set_title(f"Head {h+1}", fontsize=9)
        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=90, fontsize=6)
        ax.set_yticks(range(len(src_tokens)))
        ax.set_yticklabels(src_tokens, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("attention_heads.png", dpi=100, bbox_inches="tight")
    wandb.log({"attention_heads": wandb.Image("attention_heads.png")})
    plt.close()
    run.finish()


# ─────────────────────────────────────────────
# Experiment 4: Sinusoidal vs Learned PE
# ─────────────────────────────────────────────
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.embedding(positions))


def exp_pe_ablation(train_loader, val_loader, tgt_itos,
                    src_vocab_size, tgt_vocab_size, cfg, device, epochs=10):
    for pe_type in ["sinusoidal", "learned"]:
        run = wandb.init(project=cfg["project"], name=f"exp4_{pe_type}_pe", reinit=True,
                         config={**cfg, "pe_type": pe_type})

        model = build_model(src_vocab_size, tgt_vocab_size, cfg).to(device)

        if pe_type == "learned":
            model.encoder.pos_encoding = LearnedPositionalEncoding(
                cfg["d_model"], cfg["max_len"], cfg["dropout"]).to(device)
            model.decoder.pos_encoding = LearnedPositionalEncoding(
                cfg["d_model"], cfg["max_len"], cfg["dropout"]).to(device)

        criterion = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX, smoothing=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, cfg["d_model"], warmup_steps=cfg["warmup_steps"])
        scheduler.step()

        best_bleu = 0
        for epoch in range(epochs):
            tr_loss, _ = train_epoch(model, train_loader, optimizer, criterion, scheduler, device)
            vl_loss    = val_epoch(model, val_loader, criterion, device)
            bleu       = evaluate_bleu(model, val_loader, tgt_itos, device, max_samples=200)
            wandb.log({"train_loss": tr_loss, "val_loss": vl_loss,
                       "val_bleu": bleu, "epoch": epoch + 1})
            print(f"[Exp4 {pe_type}] Ep {epoch+1}/{epochs}  bleu={bleu:.2f}")
            if bleu > best_bleu:
                best_bleu = bleu

        run.finish()


# ─────────────────────────────────────────────
# Experiment 5: Label Smoothing
# ─────────────────────────────────────────────
def exp_label_smoothing(train_loader, val_loader, tgt_itos,
                        src_vocab_size, tgt_vocab_size, cfg, device, epochs=10):
    for smoothing in [0.0, 0.1]:
        name = f"smooth_{smoothing}"
        run = wandb.init(project=cfg["project"], name=f"exp5_{name}", reinit=True,
                         config={**cfg, "label_smoothing": smoothing})

        model = build_model(src_vocab_size, tgt_vocab_size, cfg).to(device)
        criterion = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX, smoothing=smoothing)
        optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, cfg["d_model"], warmup_steps=cfg["warmup_steps"])
        scheduler.step()

        for epoch in range(epochs):
            model.train()
            total_loss  = 0
            total_conf  = 0
            n_batches   = 0

            for src, tgt in train_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_in  = tgt[:, :-1]
                tgt_out = tgt[:, 1:]

                optimizer.zero_grad()
                logits = model(src, tgt_in)
                logits_flat  = logits.reshape(-1, logits.size(-1))
                targets_flat = tgt_out.reshape(-1)
                loss = criterion(logits_flat, targets_flat)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Prediction confidence: softmax prob of correct token
                with torch.no_grad():
                    probs = torch.softmax(logits_flat, dim=-1)
                    mask  = targets_flat != PAD_IDX
                    if mask.sum() > 0:
                        correct_probs = probs[mask].gather(1, targets_flat[mask].unsqueeze(1))
                        total_conf += correct_probs.mean().item()

                total_loss += loss.item()
                n_batches  += 1

            vl_loss = val_epoch(model, val_loader, criterion, device)
            bleu    = evaluate_bleu(model, val_loader, tgt_itos, device, max_samples=200)
            wandb.log({
                "train_loss": total_loss / n_batches,
                "val_loss":   vl_loss,
                "val_bleu":   bleu,
                "pred_confidence": total_conf / n_batches,
                "epoch": epoch + 1,
            })
            print(f"[Exp5 smooth={smoothing}] Ep {epoch+1}/{epochs}  bleu={bleu:.2f}")

        run.finish()


# ─────────────────────────────────────────────
# Main training run (saves best model)
# ─────────────────────────────────────────────
def main_train(train_loader, val_loader, test_loader, tgt_itos,
               src_vocab_size, tgt_vocab_size, cfg, device, epochs=30):
    run = wandb.init(project=cfg["project"], name="main_training", reinit=True, config=cfg)

    model = build_model(src_vocab_size, tgt_vocab_size, cfg).to(device)
    criterion = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX, smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, cfg["d_model"], warmup_steps=cfg["warmup_steps"])
    scheduler.step()   # step=1 for epoch 1

    best_bleu = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr_loss, _ = train_epoch(model, train_loader, optimizer, criterion, scheduler, device)
        vl_loss    = val_epoch(model, val_loader, criterion, device)
        bleu       = evaluate_bleu(model, val_loader, tgt_itos, device, max_samples=500)
        lr_now     = optimizer.param_groups[0]["lr"]

        wandb.log({"train_loss": tr_loss, "val_loss": vl_loss,
                   "val_bleu": bleu, "lr": lr_now, "epoch": epoch})
        print(f"Epoch {epoch:3d}/{epochs}  train={tr_loss:.4f}  val={vl_loss:.4f}"
              f"  bleu={bleu:.2f}  lr={lr_now:.2e}")

        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f"  ✓ Saved best model (BLEU={best_bleu:.2f})")

    # Test BLEU on held-out set
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    test_bleu = evaluate_bleu(model, test_loader, tgt_itos, device, max_samples=1000)
    wandb.log({"test_bleu": test_bleu})
    print(f"\n[Final] Test BLEU = {test_bleu:.2f}")
    run.finish()
    return model


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project",        default="da6401_assignment3")
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--batch_size",     type=int,   default=128)
    p.add_argument("--d_model",        type=int,   default=256)
    p.add_argument("--num_heads",      type=int,   default=8)
    p.add_argument("--num_layers",     type=int,   default=3)
    p.add_argument("--d_ff",           type=int,   default=512)
    p.add_argument("--max_len",        type=int,   default=256)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--warmup_steps",   type=int,   default=4000)
    p.add_argument("--exp_epochs",     type=int,   default=10,
                   help="Epochs for ablation experiments")
    p.add_argument("--run_experiments",action="store_true",
                   help="Run all W&B ablation experiments after main training")
    p.add_argument("--only_experiments",action="store_true",
                   help="Run only experiments (skip main training)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = dict(
        project      = args.project,
        d_model      = args.d_model,
        num_heads    = args.num_heads,
        num_layers   = args.num_layers,
        d_ff         = args.d_ff,
        max_len      = args.max_len,
        dropout      = args.dropout,
        warmup_steps = args.warmup_steps,
    )

    print("[Data] Loading Multi30k …")
    (train_loader, val_loader, test_loader,
     src_vocab, tgt_vocab, src_itos, tgt_itos) = get_dataloaders(args.batch_size, args.max_len)

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    print(f"[Data] src_vocab={src_vocab_size}  tgt_vocab={tgt_vocab_size}")

    trained_model = None
    if not args.only_experiments:
        print("\n=== Main Training ===")
        trained_model = main_train(train_loader, val_loader, test_loader, tgt_itos,
                                   src_vocab_size, tgt_vocab_size, cfg, device, args.epochs)

    if args.run_experiments or args.only_experiments:
        print("\n=== Experiment 1: Noam vs Fixed LR ===")
        exp_noam_vs_fixed(train_loader, val_loader, tgt_itos,
                          src_vocab_size, tgt_vocab_size, cfg, device, args.exp_epochs)

        print("\n=== Experiment 2: Scaling Factor Ablation ===")
        exp_scaling_ablation(train_loader, val_loader,
                             src_vocab_size, tgt_vocab_size, cfg, device)

        print("\n=== Experiment 3: Attention Rollout ===")
        if trained_model is None:
            trained_model = build_model(src_vocab_size, tgt_vocab_size, cfg).to(device)
            if os.path.isfile("checkpoints/best_model.pt"):
                trained_model.load_state_dict(
                    torch.load("checkpoints/best_model.pt", map_location=device))
        exp_attention_rollout(trained_model, val_loader, src_itos, tgt_itos, device, cfg)

        print("\n=== Experiment 4: Sinusoidal vs Learned PE ===")
        exp_pe_ablation(train_loader, val_loader, tgt_itos,
                        src_vocab_size, tgt_vocab_size, cfg, device, args.exp_epochs)

        print("\n=== Experiment 5: Label Smoothing ===")
        exp_label_smoothing(train_loader, val_loader, tgt_itos,
                            src_vocab_size, tgt_vocab_size, cfg, device, args.exp_epochs)

    print("\nDone!")
