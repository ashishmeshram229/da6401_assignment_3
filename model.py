"""
model.py - Transformer model for German-to-English Machine Translation
DA6401 Assignment 3

Implements the full Transformer architecture from "Attention Is All You Need"
including the infer() method for end-to-end inference as required by the autograder.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
import gdown
import pickle


# ─────────────────────────────────────────────
# Scaled Dot-Product Attention
# ─────────────────────────────────────────────
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.
    Q, K, V: (batch, heads, seq_len, d_k)
    mask:    (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
    Returns: output (batch, heads, seq_len, d_k), attn_weights (batch, heads, seq_len, seq_len)
    """
    d_k = Q.size(-1)
    # (batch, heads, seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    # Replace NaN from fully-masked rows with 0
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    output = torch.matmul(attn_weights, V)
    return output, attn_weights


# ─────────────────────────────────────────────
# Multi-Head Attention
# ─────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """(batch, seq, d_model) -> (batch, heads, seq, d_k)"""
        batch, seq, _ = x.size()
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))

        x, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # (batch, heads, seq, d_k) -> (batch, seq, d_model)
        batch, _, seq, _ = x.size()
        x = x.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.W_O(x)


# ─────────────────────────────────────────────
# Point-wise Feed-Forward Network
# ─────────────────────────────────────────────
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ─────────────────────────────────────────────
# Sinusoidal Positional Encoding
# ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────
# Encoder Layer
# ─────────────────────────────────────────────
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Post-LN (as in original paper)
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ─────────────────────────────────────────────
# Decoder Layer
# ─────────────────────────────────────────────
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # Masked self-attention (causal)
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        # Cross-attention
        attn2 = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn2))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


# ─────────────────────────────────────────────
# Encoder Stack
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.scale = math.sqrt(d_model)

    def forward(self, src, src_mask=None):
        x = self.pos_encoding(self.embedding(src) * self.scale)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


# ─────────────────────────────────────────────
# Decoder Stack
# ─────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.scale = math.sqrt(d_model)

    def forward(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.pos_encoding(self.embedding(tgt) * self.scale)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x


# ─────────────────────────────────────────────
# Mask Utilities
# ─────────────────────────────────────────────
def make_padding_mask(seq, pad_idx=1):
    """
    seq: (batch, seq_len)  integer token ids
    Returns: (batch, 1, 1, seq_len)  bool mask (1 = keep, 0 = mask)
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(seq_len, device):
    """
    Returns lower-triangular mask (1 = keep, 0 = mask) of shape (1, 1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def make_tgt_mask(tgt, pad_idx=1):
    """
    Combined padding + causal mask for decoder.
    tgt: (batch, seq_len)
    Returns: (batch, 1, seq_len, seq_len)
    """
    device = tgt.device
    seq_len = tgt.size(1)
    pad_mask   = make_padding_mask(tgt, pad_idx)               # (batch, 1, 1, seq_len)
    causal_mask = make_causal_mask(seq_len, device)            # (1, 1, seq_len, seq_len)
    return pad_mask & causal_mask                              # broadcasts -> (batch, 1, seq_len, seq_len)


# ─────────────────────────────────────────────
# Noam Learning-Rate Scheduler
# ─────────────────────────────────────────────
class NoamScheduler:
    """
    lrate = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    """
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._compute_lr(self._step)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def _compute_lr(self, step):
        return (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5)
        )

    def get_lr(self):
        return self._compute_lr(self._step)


# ─────────────────────────────────────────────
# Label-Smoothing Loss
# ─────────────────────────────────────────────
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, pad_idx=1, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        """
        logits: (N, vocab_size)
        target: (N,)
        """
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_dist = torch.zeros_like(log_probs)
            smooth_dist.fill_(self.smoothing / (self.vocab_size - 2))   # -2 for pad & gold
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_dist[:, self.pad_idx] = 0.0
            mask = (target == self.pad_idx)
            smooth_dist[mask] = 0.0

        loss = -(smooth_dist * log_probs).sum(dim=-1)
        # Normalise by non-pad tokens
        non_pad = (~mask).sum()
        return loss.sum() / (non_pad + 1e-9)


# ─────────────────────────────────────────────
# Full Transformer
# ─────────────────────────────────────────────
class Transformer(nn.Module):
    """
    German → English Neural Machine Translation Transformer.

    All default hyper-parameters match the "base" model from the paper.
    Vocabulary, tokenisers, and trained weights are loaded inside __init__
    as required by the autograder.
    """

    # ── Hyper-parameters (all have defaults) ───────────────────────────
    def __init__(
        self,
        src_vocab_size: int = 8500,     # set properly after vocab build; overridden at runtime
        tgt_vocab_size: int = 6500,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        d_ff: int = 512,
        max_len: int = 256,
        dropout: float = 0.1,
        pad_idx: int = 1,
        # Google Drive file ID for trained weights (set after training)
        weights_gdrive_id: str = "",
        # Local path of vocab file (relative to this file)
        vocab_path: str = "",
    ):
        super().__init__()

        # ── Load tokenisers ────────────────────────────────────────────
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            from spacy.cli import download as spacy_download
            spacy_download("de_core_news_sm")
            self.spacy_de = spacy.load("de_core_news_sm")

        try:
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download as spacy_download
            spacy_download("en_core_web_sm")
            self.spacy_en = spacy.load("en_core_web_sm")

        # ── Load / resolve vocab ───────────────────────────────────────
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Accept explicit path or search standard locations
        candidate_paths = [
            vocab_path,
            os.path.join(base_dir, "vocab.pkl"),
            os.path.join(base_dir, "checkpoints", "vocab.pkl"),
            "vocab.pkl",
            "checkpoints/vocab.pkl",
        ]

        vocab_loaded = False
        for cp in candidate_paths:
            if cp and os.path.isfile(cp):
                with open(cp, "rb") as f:
                    vocab_data = pickle.load(f)
                self.src_vocab = vocab_data["src_vocab"]
                self.tgt_vocab = vocab_data["tgt_vocab"]
                self.src_itos  = vocab_data["src_itos"]
                self.tgt_itos  = vocab_data["tgt_itos"]
                src_vocab_size = len(self.src_vocab)
                tgt_vocab_size = len(self.tgt_vocab)
                vocab_loaded = True
                break

        if not vocab_loaded:
            # Build minimal placeholder (useful during first-run / training)
            self.src_vocab: dict = {}
            self.tgt_vocab: dict = {}
            self.src_itos: list  = []
            self.tgt_itos: list  = []

        self.pad_idx    = pad_idx
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_len    = max_len

        # ── Build model components ─────────────────────────────────────
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, num_layers,
                               d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, num_layers,
                               d_ff, max_len, dropout)
        self.fc_out  = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

        # ── Load trained weights ───────────────────────────────────────
        weight_paths = [
            os.path.join(base_dir, "checkpoints", "best_model.pt"),
            os.path.join(base_dir, "best_model.pt"),
            "checkpoints/best_model.pt",
            "best_model.pt",
        ]
        weight_loaded = False
        for wp in weight_paths:
            if os.path.isfile(wp):
                self._load_weights(wp)
                weight_loaded = True
                break

        # Download from Google Drive if not found locally
        if not weight_loaded and weights_gdrive_id:
            os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
            dest = os.path.join(base_dir, "checkpoints", "best_model.pt")
            print(f"[Transformer] Downloading weights from Google Drive …")
            gdown.download(
                f"https://drive.google.com/uc?id={weights_gdrive_id}",
                dest, quiet=False
            )
            if os.path.isfile(dest):
                self._load_weights(dest)

    # ── Weight initialisation (Xavier uniform) ─────────────────────────
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _load_weights(self, path: str):
        map_loc = "cpu"
        state = torch.load(path, map_location=map_loc)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.load_state_dict(state, strict=False)
        print(f"[Transformer] Loaded weights from {path}")

    # ── Tokenise helpers ───────────────────────────────────────────────
    def _tokenize_de(self, text: str):
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def _tokenize_en(self, text: str):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    # ── Forward pass ───────────────────────────────────────────────────
    def forward(self, src, tgt):
        """
        src: (batch, src_len)  integer token ids
        tgt: (batch, tgt_len)  integer token ids  (teacher-forced)
        Returns logits: (batch, tgt_len, tgt_vocab_size)
        """
        src_mask = make_padding_mask(src, self.pad_idx)        # (batch, 1, 1, src_len)
        tgt_mask = make_tgt_mask(tgt, self.pad_idx)            # (batch, 1, tgt_len, tgt_len)

        enc_out  = self.encoder(src, src_mask)
        dec_out  = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        logits   = self.fc_out(dec_out)
        return logits

    # ── Greedy Decoding ────────────────────────────────────────────────
    def greedy_decode(self, src, src_mask, max_decode_len=50):
        """
        src:      (1, src_len) integer tensor
        src_mask: (1, 1, 1, src_len)
        Returns a list of integer token ids (without BOS).
        """
        device = src.device
        sos_idx = self.tgt_vocab.get("<sos>", 2)
        eos_idx = self.tgt_vocab.get("<eos>", 3)

        enc_out = self.encoder(src, src_mask)
        tgt = torch.tensor([[sos_idx]], device=device)

        for _ in range(max_decode_len):
            tgt_mask = make_causal_mask(tgt.size(1), device)
            dec_out  = self.decoder(tgt, enc_out, src_mask, tgt_mask)
            logits   = self.fc_out(dec_out[:, -1, :])          # last position
            next_tok = logits.argmax(dim=-1, keepdim=True)     # greedy

            if next_tok.item() == eos_idx:
                break
            tgt = torch.cat([tgt, next_tok], dim=1)

        return tgt[0, 1:].tolist()   # strip BOS

    # ── End-to-end inference (required by autograder) ──────────────────
    def infer(self, german_sentence: str, max_decode_len: int = 50) -> str:
        """
        Accept a German sentence (string), return the English translation (string).
        This method is called by the autograder as:
            english_sentence = model.infer(german_sentence)
        """
        self.eval()
        device = next(self.parameters()).device

        # ── Tokenise German sentence ───────────────────────────────────
        tokens = self._tokenize_de(german_sentence)

        sos_idx = self.src_vocab.get("<sos>", 2)
        eos_idx = self.src_vocab.get("<eos>", 3)
        unk_idx = self.src_vocab.get("<unk>", 0)

        ids = [sos_idx] + [self.src_vocab.get(t, unk_idx) for t in tokens] + [eos_idx]
        src = torch.tensor([ids], dtype=torch.long, device=device)  # (1, src_len)
        src_mask = make_padding_mask(src, self.pad_idx)              # (1,1,1,src_len)

        with torch.no_grad():
            pred_ids = self.greedy_decode(src, src_mask, max_decode_len)

        # ── Detokenise ────────────────────────────────────────────────
        eos_idx_tgt = self.tgt_vocab.get("<eos>", 3)
        words = []
        for idx in pred_ids:
            if idx == eos_idx_tgt:
                break
            token = self.tgt_itos[idx] if idx < len(self.tgt_itos) else "<unk>"
            if token not in ("<sos>", "<eos>", "<pad>", "<unk>"):
                words.append(token)

        return " ".join(words)
