import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"


# Google Drive file id for the trained checkpoint.
DEFAULT_GOOGLE_DRIVE_FILE_ID = os.environ.get("DA6401_WEIGHT_FILE_ID", "1FtCkmIFWONDXQ6eR43dbGa-WIQ8RkBSH")


def get_default_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def normalize_drive_id(file_id_or_url: str) -> str:
    if not file_id_or_url:
        return ""

    text = file_id_or_url.strip()
    patterns = [
        r"/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"file_id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return text


class SimpleVocab:
    def __init__(self, stoi: Dict[str, int], itos: Optional[List[str]] = None):
        self.stoi = {str(token): int(index) for token, index in stoi.items()}

        if itos is None:
            itos = [""] * (max(self.stoi.values()) + 1)
            for token, index in self.stoi.items():
                itos[index] = token

        self.itos = [str(token) for token in itos]
        self.pad_idx = self.stoi.get(PAD_TOKEN, 0)
        self.unk_idx = self.stoi.get(UNK_TOKEN, self.pad_idx)
        self.sos_idx = self.stoi.get(SOS_TOKEN, self.stoi.get("<bos>", 1))
        self.eos_idx = self.stoi.get(EOS_TOKEN, self.stoi.get("<eos>", 2))

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: List[str], add_special_tokens: bool = True) -> List[int]:
        ids = [self.stoi.get(token, self.unk_idx) for token in tokens]
        if add_special_tokens:
            ids = [self.sos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, ids: List[int]) -> str:
        words = []
        for idx in ids:
            if idx < 0 or idx >= len(self.itos):
                continue
            token = self.itos[idx]
            if token in {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, "<bos>", "<eos>"}:
                continue
            words.append(token)
        return clean_translation(" ".join(words))


def load_vocab(path: str) -> SimpleVocab:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "stoi" in data:
        return SimpleVocab(data["stoi"], data.get("itos"))

    if isinstance(data, dict):
        return SimpleVocab(data)

    if isinstance(data, list):
        return SimpleVocab({token: i for i, token in enumerate(data)}, data)

    raise ValueError(f"Unsupported vocab format in {path}")


def vocab_from_data(data) -> Optional[SimpleVocab]:
    if data is None:
        return None
    if isinstance(data, SimpleVocab):
        return data
    if isinstance(data, dict) and "stoi" in data:
        return SimpleVocab(data["stoi"], data.get("itos"))
    if isinstance(data, dict):
        return SimpleVocab(data)
    if isinstance(data, list):
        return SimpleVocab({token: i for i, token in enumerate(data)}, data)
    return None


def fallback_vocab() -> SimpleVocab:
    tokens = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
    return SimpleVocab({token: i for i, token in enumerate(tokens)}, tokens)


def clean_translation(text: str) -> str:
    text = text.replace(" n't", "n't")
    text = text.replace(" 's", "'s")
    text = text.replace(" 're", "'re")
    text = text.replace(" 'm", "'m")
    text = text.replace(" 've", "'ve")
    text = text.replace(" 'll", "'ll")
    text = re.sub(r"\s+([.,!?;:%])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def mask_to_keep(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        flat = mask.reshape(-1)
        if flat.numel() > 0 and not bool(flat[0].item()):
            return ~mask
        return mask

    flat = mask.reshape(-1)
    if flat.numel() > 0 and float(flat[0].item()) == 0.0:
        return mask == 0
    return mask != 0


class SpacyTokenizer:
    def __init__(self, language: str):
        self.language = language
        self.nlp = None

        try:
            import spacy

            if language == "de":
                try:
                    self.nlp = spacy.load("de_core_news_sm")
                except OSError:
                    self.nlp = spacy.blank("de")
            elif language == "en":
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    self.nlp = spacy.blank("en")
            else:
                self.nlp = spacy.blank(language)
        except Exception:
            self.nlp = None

    def __call__(self, text: str) -> List[str]:
        text = text.strip().lower()
        if not text:
            return []

        if self.nlp is None:
            return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

        return [token.text.lower() for token in self.nlp.tokenizer(text)]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1, use_scaling: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.use_scaling = use_scaling

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1))

        if self.use_scaling:
            scores = scores / math.sqrt(query.size(-1))

        if mask is not None:
            mask = mask_to_keep(mask)
            if mask.dim() == 2:
                if mask.size(0) == query.size(0) and mask.size(1) == key.size(-2):
                    mask = mask.unsqueeze(1).unsqueeze(2)
                else:
                    mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.to(device=scores.device)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, value)
        return output, attention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout, use_scaling=use_scaling)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def get_query_mask(
        self,
        mask: Optional[torch.Tensor],
        query_len: int,
        key_len: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None

        keep_mask = mask_to_keep(mask)

        if keep_mask.dim() == 2 and keep_mask.size(0) == batch_size:
            if keep_mask.size(1) == query_len:
                return keep_mask.to(device=device, dtype=dtype).unsqueeze(-1)

        if keep_mask.dim() == 3 and keep_mask.size(0) == batch_size:
            if query_len == key_len and keep_mask.size(1) == query_len and keep_mask.size(2) == key_len:
                columns = keep_mask.to(device=device, dtype=dtype).sum(dim=1) > 0
                return columns.to(dtype=dtype).unsqueeze(-1)
            if keep_mask.size(1) == query_len and keep_mask.size(2) == key_len:
                rows = keep_mask.to(device=device, dtype=dtype).sum(dim=-1) > 0
                return rows.to(dtype=dtype).unsqueeze(-1)
            if keep_mask.size(1) == 1 and keep_mask.size(2) == query_len:
                return keep_mask[:, 0, :].to(device=device, dtype=dtype).unsqueeze(-1)

        if keep_mask.dim() == 4 and keep_mask.size(0) == batch_size:
            if query_len == key_len and keep_mask.size(-2) == query_len and keep_mask.size(-1) == key_len:
                columns = keep_mask.to(device=device, dtype=dtype).sum(dim=1).sum(dim=-2) > 0
                return columns.to(dtype=dtype).unsqueeze(-1)
            if keep_mask.size(-2) == query_len and keep_mask.size(-1) == key_len:
                rows = keep_mask.to(device=device, dtype=dtype).sum(dim=1).sum(dim=-1) > 0
                return rows.to(dtype=dtype).unsqueeze(-1)
            if keep_mask.size(-2) == 1 and keep_mask.size(-1) == query_len:
                return keep_mask[:, 0, 0, :].to(device=device, dtype=dtype).unsqueeze(-1)

        return None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        query_mask = self.get_query_mask(
            mask,
            query_len=query.size(1),
            key_len=key.size(1),
            batch_size=query.size(0),
            device=query.device,
            dtype=query.dtype,
        )

        q = self.split_heads(self.w_q(query))
        k = self.split_heads(self.w_k(key))
        v = self.split_heads(self.w_v(value))

        output, attention = self.attention(q, k, v, mask)
        self.attention_weights = attention
        output = self.combine_heads(output)
        output = self.fc_out(output)
        output = self.dropout(output)

        if query_mask is not None:
            output = output * query_mask

        if return_attention:
            return output, attention
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 256, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)].to(x.device)
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 256, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        return self.dropout(x + self.embedding(positions))


class FeedForward(nn.Module):
    def __init__(self, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_output, attention = self.self_attention(src, src, src, src_mask, return_attention=True)
        src = self.norm1(src + self.dropout(attention_output))

        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))
        return src, attention


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self_output, self_attention = self.self_attention(tgt, tgt, tgt, tgt_mask, return_attention=True)
        tgt = self.norm1(tgt + self.dropout(self_output))

        cross_output, cross_attention = self.cross_attention(
            tgt, memory, memory, src_mask, return_attention=True
        )
        tgt = self.norm2(tgt + self.dropout(cross_output))

        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_output))
        return tgt, self_attention, cross_attention


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_idx: int = 0,
        positional_encoding: str = "sinusoidal",
        use_scaling: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        if positional_encoding == "learned":
            self.position_encoding = LearnedPositionalEncoding(d_model, max_len, dropout)
        else:
            self.position_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout, use_scaling) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        src = self.token_embedding(src) * math.sqrt(self.d_model)
        src = self.position_encoding(src)

        attention_maps = []
        for layer in self.layers:
            src, attention = layer(src, src_mask)
            attention_maps.append(attention)

        return src, attention_maps


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_idx: int = 0,
        positional_encoding: str = "sinusoidal",
        use_scaling: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        if positional_encoding == "learned":
            self.position_encoding = LearnedPositionalEncoding(d_model, max_len, dropout)
        else:
            self.position_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout, use_scaling) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        tgt = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.position_encoding(tgt)

        self_attention_maps = []
        cross_attention_maps = []

        for layer in self.layers:
            tgt, self_attention, cross_attention = layer(tgt, memory, tgt_mask, src_mask)
            self_attention_maps.append(self_attention)
            cross_attention_maps.append(cross_attention)

        logits = self.fc_out(tgt)
        return logits, self_attention_maps, cross_attention_maps


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 5000,
        src_pad_idx: Optional[int] = None,
        tgt_pad_idx: Optional[int] = None,
        positional_encoding: str = "sinusoidal",
        use_scaling: bool = True,
        src_vocab_path: Optional[str] = None,
        tgt_vocab_path: Optional[str] = None,
        weight_path: Optional[str] = None,
        google_drive_file_id: str = DEFAULT_GOOGLE_DRIVE_FILE_ID,
        load_weights: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        requested_src_vocab_size = src_vocab_size
        requested_tgt_vocab_size = tgt_vocab_size
        self.weights_loaded = False
        self.device_name = device if device is not None else get_default_device()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len

        self.src_tokenizer = SpacyTokenizer("de")
        self.tgt_tokenizer = SpacyTokenizer("en")

        src_vocab_path = src_vocab_path or os.path.join(base_dir, "vocab", "src_vocab.json")
        tgt_vocab_path = tgt_vocab_path or os.path.join(base_dir, "vocab", "tgt_vocab.json")
        self.weight_path = weight_path or os.path.join(base_dir, "checkpoints", "transformer_best.pt")

        if load_weights:
            self._download_weights_if_needed(google_drive_file_id)

        checkpoint = self._read_checkpoint() if load_weights else None
        checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

        d_model = checkpoint_config.get("d_model", d_model)
        num_heads = checkpoint_config.get("num_heads", num_heads)
        num_layers = checkpoint_config.get("num_layers", num_layers)
        d_ff = checkpoint_config.get("d_ff", d_ff)
        dropout = checkpoint_config.get("dropout", dropout)
        max_len = checkpoint_config.get("max_len", max_len)
        positional_encoding = checkpoint_config.get("positional_encoding", positional_encoding)
        use_scaling = checkpoint_config.get("use_scaling", use_scaling)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len

        self.src_vocab = load_vocab(src_vocab_path) if os.path.exists(src_vocab_path) else None
        self.tgt_vocab = load_vocab(tgt_vocab_path) if os.path.exists(tgt_vocab_path) else None

        if checkpoint is not None and isinstance(checkpoint, dict):
            self.src_vocab = self.src_vocab or vocab_from_data(checkpoint.get("src_vocab"))
            self.tgt_vocab = self.tgt_vocab or vocab_from_data(checkpoint.get("tgt_vocab"))

        if self.src_vocab is None and requested_src_vocab_size is None:
            self.src_vocab = fallback_vocab()
        if self.tgt_vocab is None and requested_tgt_vocab_size is None:
            self.tgt_vocab = fallback_vocab()

        if self.src_vocab is not None:
            src_vocab_size = len(self.src_vocab)
            src_pad_idx = self.src_vocab.pad_idx

        if self.tgt_vocab is not None:
            tgt_vocab_size = len(self.tgt_vocab)
            tgt_pad_idx = self.tgt_vocab.pad_idx

        state_dict = self._state_dict_from_checkpoint(checkpoint)
        if state_dict is not None:
            src_weight = state_dict.get("encoder.token_embedding.weight")
            tgt_weight = state_dict.get("decoder.token_embedding.weight")
            if src_weight is not None:
                src_vocab_size = src_weight.size(0)
            if tgt_weight is not None:
                tgt_vocab_size = tgt_weight.size(0)

        src_vocab_size = src_vocab_size or 12000
        tgt_vocab_size = tgt_vocab_size or 12000
        self.src_pad_idx = 0 if src_pad_idx is None else src_pad_idx
        self.tgt_pad_idx = 0 if tgt_pad_idx is None else tgt_pad_idx

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
            pad_idx=self.src_pad_idx,
            positional_encoding=positional_encoding,
            use_scaling=use_scaling,
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
            pad_idx=self.tgt_pad_idx,
            positional_encoding=positional_encoding,
            use_scaling=use_scaling,
        )

        self.to(self.device_name)

        if load_weights:
            self._load_weights_if_available(checkpoint)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        batch_size, tgt_len = tgt.size()
        padding_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        return padding_mask & causal_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        memory, encoder_attention = self.encoder(src, src_mask)
        logits, decoder_self_attention, cross_attention = self.decoder(tgt, memory, tgt_mask, src_mask)

        attention_maps = {
            "encoder": encoder_attention,
            "decoder_self": decoder_self_attention,
            "decoder_cross": cross_attention,
        }
        return logits, attention_maps

    def encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        src_mask = self.make_src_mask(src)
        memory, attention = self.encoder(src, src_mask)
        return memory, src_mask, attention

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        tgt_mask = self.make_tgt_mask(tgt)
        logits, self_attention, cross_attention = self.decoder(tgt, memory, tgt_mask, src_mask)
        return logits, {"decoder_self": self_attention, "decoder_cross": cross_attention}

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, max_len: int = 80) -> Tuple[List[int], Dict[str, List[torch.Tensor]]]:
        self.eval()
        src = src.to(next(self.parameters()).device)
        memory, src_mask, encoder_attention = self.encode(src)

        if self.tgt_vocab is None:
            raise RuntimeError("Target vocabulary is missing. Add vocab/tgt_vocab.json before inference.")

        generated = torch.tensor([[self.tgt_vocab.sos_idx]], device=src.device, dtype=torch.long)
        last_attention = {"encoder": encoder_attention, "decoder_self": [], "decoder_cross": []}

        for _ in range(max_len - 1):
            logits, decoder_attention = self.decode(generated, memory, src_mask)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            last_attention["decoder_self"] = decoder_attention["decoder_self"]
            last_attention["decoder_cross"] = decoder_attention["decoder_cross"]

            if next_token.item() == self.tgt_vocab.eos_idx:
                break

        return generated.squeeze(0).tolist(), last_attention

    @torch.no_grad()
    def beam_search_decode(
        self,
        src: torch.Tensor,
        max_len: int = 80,
        beam_size: int = 4,
        length_penalty: float = 0.6,
    ) -> List[int]:
        self.eval()
        src = src.to(next(self.parameters()).device)
        memory, src_mask, _ = self.encode(src)

        if self.tgt_vocab is None:
            raise RuntimeError("Target vocabulary is missing. Add vocab/tgt_vocab.json before inference.")

        start = self.tgt_vocab.sos_idx
        end = self.tgt_vocab.eos_idx
        beams = [(torch.tensor([[start]], device=src.device, dtype=torch.long), 0.0)]
        finished = []

        for _ in range(max_len - 1):
            candidates = []

            for tokens, score in beams:
                if tokens[0, -1].item() == end:
                    finished.append((tokens, score))
                    candidates.append((tokens, score))
                    continue

                logits, _ = self.decode(tokens, memory, src_mask)
                log_probs = F.log_softmax(logits[:, -1], dim=-1)
                top_scores, top_ids = torch.topk(log_probs, beam_size, dim=-1)

                for i in range(beam_size):
                    next_id = top_ids[0, i].view(1, 1)
                    next_score = score + top_scores[0, i].item()
                    next_tokens = torch.cat([tokens, next_id], dim=1)
                    candidates.append((next_tokens, next_score))

            def normalized(item):
                tokens, score = item
                length = max(tokens.size(1) - 1, 1)
                penalty = ((5.0 + length) / 6.0) ** length_penalty
                return score / penalty

            beams = sorted(candidates, key=normalized, reverse=True)[:beam_size]

            if all(tokens[0, -1].item() == end for tokens, _ in beams):
                break

        finished.extend(beams)
        best_tokens, _ = max(finished, key=lambda item: item[1] / (((5.0 + max(item[0].size(1) - 1, 1)) / 6.0) ** length_penalty))
        return best_tokens.squeeze(0).tolist()

    @torch.no_grad()
    def infer(self, german_sentence: str, max_len: int = 80, beam_size: int = 4) -> str:
        if not self.weights_loaded:
            raise RuntimeError(
                "Trained weights were not loaded. Set DEFAULT_GOOGLE_DRIVE_FILE_ID "
                "or include checkpoints/transformer_best.pt before running infer()."
            )
        if self.src_vocab is None:
            raise RuntimeError("Source vocabulary is missing. Add vocab/src_vocab.json before inference.")
        if self.tgt_vocab is None:
            raise RuntimeError("Target vocabulary is missing. Add vocab/tgt_vocab.json before inference.")

        tokens = self.src_tokenizer(german_sentence)
        ids = self.src_vocab.encode(tokens, add_special_tokens=True)
        src = torch.tensor(ids, dtype=torch.long, device=next(self.parameters()).device).unsqueeze(0)

        if beam_size and beam_size > 1:
            output_ids = self.beam_search_decode(src, max_len=max_len, beam_size=beam_size)
        else:
            output_ids, _ = self.greedy_decode(src, max_len=max_len)
        return self.tgt_vocab.decode(output_ids)

    def get_attention_rollout(self, attention_maps: List[torch.Tensor]) -> torch.Tensor:
        if not attention_maps:
            raise ValueError("attention_maps is empty")

        rollout = None
        for attention in attention_maps:
            attention = attention.detach()
            attention = attention.mean(dim=1)
            eye = torch.eye(attention.size(-1), device=attention.device).unsqueeze(0)
            attention = attention + eye
            attention = attention / attention.sum(dim=-1, keepdim=True)

            rollout = attention if rollout is None else torch.matmul(attention, rollout)

        return rollout

    def _download_weights_if_needed(self, google_drive_file_id: str) -> None:
        google_drive_file_id = normalize_drive_id(google_drive_file_id)
        if not google_drive_file_id:
            return

        if os.path.exists(self.weight_path):
            return

        os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)

        try:
            import gdown
        except ImportError as exc:
            raise ImportError("gdown is required for downloading pretrained weights.") from exc

        gdown.download(id=google_drive_file_id, output=self.weight_path, quiet=False)

    def _read_checkpoint(self):
        if not os.path.exists(self.weight_path):
            return None

        return torch.load(self.weight_path, map_location=self.device_name)

    def _state_dict_from_checkpoint(self, checkpoint):
        if checkpoint is None:
            return None
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint

    def _load_weights_if_available(self, checkpoint=None) -> None:
        checkpoint = checkpoint if checkpoint is not None else self._read_checkpoint()
        state_dict = self._state_dict_from_checkpoint(checkpoint)
        if state_dict is None:
            return

        self.load_state_dict(state_dict, strict=True)
        self.to(self.device_name)
        self.weights_loaded = True


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size)
        target = target.reshape(-1)

        mask = target != self.ignore_index
        logits = logits[mask]
        target = target[mask]

        if target.numel() == 0:
            return logits.sum() * 0.0

        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
