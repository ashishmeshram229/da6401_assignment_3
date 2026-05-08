"""
dataset.py - Data loading, vocabulary building, and batching utilities
DA6401 Assignment 3
"""

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import spacy
from datasets import load_dataset


# ─────────────────────────────────────────────
# Tokenizers
# ─────────────────────────────────────────────
def get_tokenizers():
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except OSError:
        from spacy.cli import download
        download("de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    def tokenize_de(text):
        return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

    return tokenize_de, tokenize_en


# ─────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────
SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3


def build_vocab(sentences, tokenize_fn, min_freq=2):
    """Build vocab from list of sentences."""
    from collections import Counter
    counter = Counter()
    for sent in sentences:
        counter.update(tokenize_fn(sent))

    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for tok, freq in counter.items():
        if freq >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)

    itos = {idx: tok for tok, idx in vocab.items()}
    itos_list = [itos[i] for i in range(len(itos))]
    return vocab, itos_list


def save_vocab(src_vocab, tgt_vocab, src_itos, tgt_itos, path="checkpoints/vocab.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab,
            "src_itos": src_itos,
            "tgt_itos": tgt_itos,
        }, f)
    print(f"[Vocab] Saved to {path}")


def load_vocab(path="checkpoints/vocab.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["src_vocab"], data["tgt_vocab"], data["src_itos"], data["tgt_itos"]


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class Multi30kDataset(Dataset):
    def __init__(self, split, src_vocab, tgt_vocab, tokenize_de, tokenize_en, max_len=256):
        super().__init__()
        raw = load_dataset("bentrevett/multi30k", split=split)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tokenize_de = tokenize_de
        self.tokenize_en = tokenize_en
        self.max_len = max_len

        self.data = []
        for item in raw:
            de = item["de"]
            en = item["en"]
            de_toks = tokenize_de(de)[:max_len - 2]
            en_toks = tokenize_en(en)[:max_len - 2]
            src_ids = [SOS_IDX] + [src_vocab.get(t, UNK_IDX) for t in de_toks] + [EOS_IDX]
            tgt_ids = [SOS_IDX] + [tgt_vocab.get(t, UNK_IDX) for t in en_toks] + [EOS_IDX]
            self.data.append((torch.tensor(src_ids, dtype=torch.long),
                               torch.tensor(tgt_ids, dtype=torch.long)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


def get_dataloaders(batch_size=128, max_len=256):
    tokenize_de, tokenize_en = get_tokenizers()

    # Load raw data for vocab building
    raw_train = load_dataset("bentrevett/multi30k", split="train")
    de_sents = [item["de"] for item in raw_train]
    en_sents = [item["en"] for item in raw_train]

    src_vocab, src_itos = build_vocab(de_sents, tokenize_de, min_freq=2)
    tgt_vocab, tgt_itos = build_vocab(en_sents, tokenize_en, min_freq=2)

    save_vocab(src_vocab, tgt_vocab, src_itos, tgt_itos, path="checkpoints/vocab.pkl")
    print(f"[Vocab] German: {len(src_vocab)}, English: {len(tgt_vocab)}")

    train_ds = Multi30kDataset("train", src_vocab, tgt_vocab, tokenize_de, tokenize_en, max_len)
    val_ds   = Multi30kDataset("validation", src_vocab, tgt_vocab, tokenize_de, tokenize_en, max_len)
    test_ds  = Multi30kDataset("test", src_vocab, tgt_vocab, tokenize_de, tokenize_en, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab, src_itos, tgt_itos
