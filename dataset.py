import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, DownloadConfig, load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN, UNK_TOKEN, SimpleVocab


def build_tokenizer(language: str):
    try:
        import spacy

        if language == "de":
            try:
                nlp = spacy.load("de_core_news_sm")
            except OSError:
                nlp = spacy.blank("de")
        elif language == "en":
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                nlp = spacy.blank("en")
        else:
            nlp = spacy.blank(language)

        def tokenize(text: str) -> List[str]:
            return [token.text.lower() for token in nlp.tokenizer(text.strip().lower())]

        return tokenize

    except Exception:

        def tokenize(text: str) -> List[str]:
            return re.findall(r"\w+|[^\w\s]", text.strip().lower(), flags=re.UNICODE)

        return tokenize


def get_sentence(example: Dict, language: str) -> str:
    if language in example:
        return example[language]

    if "translation" in example and language in example["translation"]:
        return example["translation"][language]

    raise KeyError(f"Could not find language '{language}' in example keys: {list(example.keys())}")


def load_multi30k(keep_in_memory: bool = False):
    cached = load_cached_multi30k()
    if cached is not None:
        return cached

    candidates = [
        ("bentrevett/multi30k", None),
        ("multi30k", None),
    ]

    last_error = None
    for name, config in candidates:
        try:
            if config is None:
                return load_dataset(name, keep_in_memory=keep_in_memory)
            return load_dataset(name, config, keep_in_memory=keep_in_memory)
        except Exception as exc:
            last_error = exc
            try:
                download_config = DownloadConfig(local_files_only=True)
                if config is None:
                    return load_dataset(name, download_config=download_config, keep_in_memory=keep_in_memory)
                return load_dataset(name, config, download_config=download_config, keep_in_memory=keep_in_memory)
            except Exception:
                pass

    raise RuntimeError("Could not load Multi30k from HuggingFace.") from last_error


def load_cached_multi30k():
    cache_root = os.path.expanduser(
        "~/.cache/huggingface/datasets/bentrevett___multi30k/default/0.0.0"
    )
    if not os.path.exists(cache_root):
        return None

    for root, _, files in os.walk(cache_root):
        needed = {
            "train": "multi30k-train.arrow",
            "validation": "multi30k-validation.arrow",
            "test": "multi30k-test.arrow",
        }
        if all(filename in files for filename in needed.values()):
            return DatasetDict(
                {
                    split: HFDataset.from_file(os.path.join(root, filename))
                    for split, filename in needed.items()
                }
            )

    return None


def build_vocab(
    split,
    language: str,
    tokenizer,
    min_freq: int = 2,
    max_size: int = 12000,
) -> SimpleVocab:
    counter = Counter()

    for example in tqdm(split, desc=f"Building {language} vocab"):
        sentence = get_sentence(example, language)
        counter.update(tokenizer(sentence))

    special_tokens = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
    stoi = {token: i for i, token in enumerate(special_tokens)}

    for token, count in counter.most_common():
        if count < min_freq:
            continue
        if token in stoi:
            continue
        if len(stoi) >= max_size:
            break
        stoi[token] = len(stoi)

    itos = [""] * len(stoi)
    for token, index in stoi.items():
        itos[index] = token

    return SimpleVocab(stoi, itos)


def save_vocab(vocab: SimpleVocab, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"stoi": vocab.stoi, "itos": vocab.itos}, f, ensure_ascii=False, indent=2)


class TranslationDataset(Dataset):
    def __init__(
        self,
        split,
        src_vocab: SimpleVocab,
        tgt_vocab: SimpleVocab,
        src_tokenizer,
        tgt_tokenizer,
        max_len: int = 100,
    ):
        self.examples = split
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        german = get_sentence(example, "de")
        english = get_sentence(example, "en")

        src_tokens = self.src_tokenizer(german)[: self.max_len - 2]
        tgt_tokens = self.tgt_tokenizer(english)[: self.max_len - 2]

        src_ids = self.src_vocab.encode(src_tokens, add_special_tokens=True)
        tgt_ids = self.tgt_vocab.encode(tgt_tokens, add_special_tokens=True)

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_batch(batch, src_pad_idx: int, tgt_pad_idx: int):
    src_batch, tgt_batch = zip(*batch)

    src = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
    tgt = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_idx)

    return src, tgt


def make_dataloaders(
    batch_size: int = 64,
    min_freq: int = 2,
    max_vocab_size: int = 12000,
    max_len: int = 100,
    num_workers: int = 0,
    vocab_dir: str = "vocab",
    keep_in_memory: bool = False,
    pin_memory: bool = False,
):
    raw_data = load_multi30k(keep_in_memory=keep_in_memory)

    train_split = raw_data["train"]
    valid_key = "validation" if "validation" in raw_data else "valid"
    valid_split = raw_data[valid_key]
    test_split = raw_data["test"] if "test" in raw_data else None

    src_tokenizer = build_tokenizer("de")
    tgt_tokenizer = build_tokenizer("en")

    src_vocab_path = os.path.join(vocab_dir, "src_vocab.json")
    tgt_vocab_path = os.path.join(vocab_dir, "tgt_vocab.json")

    if os.path.exists(src_vocab_path):
        from model import load_vocab

        src_vocab = load_vocab(src_vocab_path)
    else:
        src_vocab = build_vocab(train_split, "de", src_tokenizer, min_freq, max_vocab_size)
        save_vocab(src_vocab, src_vocab_path)

    if os.path.exists(tgt_vocab_path):
        from model import load_vocab

        tgt_vocab = load_vocab(tgt_vocab_path)
    else:
        tgt_vocab = build_vocab(train_split, "en", tgt_tokenizer, min_freq, max_vocab_size)
        save_vocab(tgt_vocab, tgt_vocab_path)

    train_dataset = TranslationDataset(train_split, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, max_len)
    valid_dataset = TranslationDataset(valid_split, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, max_len)
    test_dataset = None
    if test_split is not None:
        test_dataset = TranslationDataset(test_split, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: collate_batch(batch, src_vocab.pad_idx, tgt_vocab.pad_idx),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: collate_batch(batch, src_vocab.pad_idx, tgt_vocab.pad_idx),
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=lambda batch: collate_batch(batch, src_vocab.pad_idx, tgt_vocab.pad_idx),
        )

    return train_loader, valid_loader, test_loader, src_vocab, tgt_vocab
