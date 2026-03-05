from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset


SPECIAL_TOKENS = {
    "doc": "<doc>",
    "pad": "<pad>",
    "bos": "<bos>",
    "eos": "<eos>",
}


@dataclass
class StreamArtifacts:
    train_path: Path
    val_path: Path
    train_tokens: int
    val_tokens: int
    manifest_path: Path
    doc_count: int


@dataclass
class DatasetBundle:
    tokenizer: Tokenizer
    train_dataset: "TokenBlockDataset"
    val_dataset: "TokenBlockDataset"
    pad_token_id: int
    doc_token_id: int
    bos_token_id: int
    eos_token_id: int
    vocab_size: int
    artifacts: StreamArtifacts


def list_text_files(corpus_dirs: Sequence[Path]) -> List[Path]:
    paths: List[Path] = []
    for corpus_dir in corpus_dirs:
        if not corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
        paths.extend(sorted(corpus_dir.glob("*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {', '.join(str(p) for p in corpus_dirs)}")
    return paths


def _tokenize_document(tokenizer: Tokenizer, text: str, doc_token: str) -> List[int]:
    prefixed = f"{doc_token} {text.strip()}"
    return tokenizer.encode(prefixed, add_special_tokens=False).ids


def _split_document_tokens(tokens: List[int], train_fraction: float) -> Tuple[List[int], List[int]]:
    if not tokens:
        return [], []
    if len(tokens) == 1:
        return tokens, []
    split_idx = int(len(tokens) * train_fraction)
    split_idx = max(1, min(split_idx, len(tokens) - 1))
    return tokens[:split_idx], tokens[split_idx:]


def _compute_fingerprint(
    doc_paths: Sequence[Path],
    tokenizer_path: Path,
    train_fraction: float,
    seed: int,
    max_documents: Optional[int],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(tokenizer_path.resolve().as_posix().encode("utf-8"))
    hasher.update(tokenizer_path.read_bytes())
    hasher.update(str(train_fraction).encode("utf-8"))
    hasher.update(str(seed).encode("utf-8"))
    hasher.update(str(max_documents).encode("utf-8"))
    for path in doc_paths:
        stat = path.stat()
        hasher.update(path.resolve().as_posix().encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
    return hasher.hexdigest()


def _write_streams(
    tokenizer: Tokenizer,
    doc_paths: Sequence[Path],
    train_fraction: float,
    doc_token: str,
    train_path: Path,
    val_path: Path,
) -> Tuple[int, int]:
    train_total = 0
    val_total = 0
    for path in doc_paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        token_ids = _tokenize_document(tokenizer, text, doc_token=doc_token)
        train_ids, val_ids = _split_document_tokens(token_ids, train_fraction=train_fraction)
        train_total += len(train_ids)
        val_total += len(val_ids)

    if train_total < 1024:
        raise RuntimeError(
            f"Train token stream too small ({train_total} tokens). Need at least 1024 tokens."
        )
    if val_total < 1024:
        raise RuntimeError(
            f"Validation token stream too small ({val_total} tokens). Need at least 1024 tokens."
        )

    train_mm = np.memmap(train_path, dtype=np.int32, mode="w+", shape=(train_total,))
    val_mm = np.memmap(val_path, dtype=np.int32, mode="w+", shape=(val_total,))

    train_pos = 0
    val_pos = 0
    for path in doc_paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        token_ids = _tokenize_document(tokenizer, text, doc_token=doc_token)
        train_ids, val_ids = _split_document_tokens(token_ids, train_fraction=train_fraction)
        if train_ids:
            chunk = np.asarray(train_ids, dtype=np.int32)
            train_mm[train_pos : train_pos + len(chunk)] = chunk
            train_pos += len(chunk)
        if val_ids:
            chunk = np.asarray(val_ids, dtype=np.int32)
            val_mm[val_pos : val_pos + len(chunk)] = chunk
            val_pos += len(chunk)

    train_mm.flush()
    val_mm.flush()
    return train_total, val_total


def prepare_token_streams(
    tokenizer: Tokenizer,
    tokenizer_path: Path,
    corpus_dirs: Sequence[Path],
    cache_dir: Path,
    cache_name: str,
    *,
    train_fraction: float = 0.9,
    seed: int = 1337,
    max_documents: Optional[int] = None,
    force_rebuild: bool = False,
    doc_token: str = SPECIAL_TOKENS["doc"],
) -> StreamArtifacts:
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = cache_dir / f"{cache_name}_train.bin"
    val_path = cache_dir / f"{cache_name}_val.bin"
    manifest_path = cache_dir / f"{cache_name}_manifest.json"

    doc_paths = list_text_files(corpus_dirs)
    rng = random.Random(seed)
    rng.shuffle(doc_paths)
    if max_documents is not None:
        doc_paths = doc_paths[:max_documents]

    fingerprint = _compute_fingerprint(
        doc_paths=doc_paths,
        tokenizer_path=tokenizer_path,
        train_fraction=train_fraction,
        seed=seed,
        max_documents=max_documents,
    )

    if not force_rebuild and train_path.exists() and val_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == fingerprint:
            return StreamArtifacts(
                train_path=train_path,
                val_path=val_path,
                train_tokens=int(manifest["train_tokens"]),
                val_tokens=int(manifest["val_tokens"]),
                manifest_path=manifest_path,
                doc_count=int(manifest["doc_count"]),
            )

    train_tokens, val_tokens = _write_streams(
        tokenizer=tokenizer,
        doc_paths=doc_paths,
        train_fraction=train_fraction,
        doc_token=doc_token,
        train_path=train_path,
        val_path=val_path,
    )

    manifest = {
        "fingerprint": fingerprint,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "doc_count": len(doc_paths),
        "train_fraction": train_fraction,
        "seed": seed,
        "max_documents": max_documents,
        "tokenizer_path": str(tokenizer_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return StreamArtifacts(
        train_path=train_path,
        val_path=val_path,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        manifest_path=manifest_path,
        doc_count=len(doc_paths),
    )


class TokenBlockDataset(Dataset):
    def __init__(
        self,
        tokens_path: Path,
        *,
        context_length: int = 1024,
        pad_token_id: Optional[int] = None,
        include_remainder: bool = False,
    ):
        self.tokens_path = tokens_path
        self.context_length = int(context_length)
        self.pad_token_id = pad_token_id
        self.include_remainder = include_remainder
        self._tokens = np.memmap(tokens_path, dtype=np.int32, mode="r")

        self._full_blocks = len(self._tokens) // self.context_length
        remainder = len(self._tokens) % self.context_length
        self._has_remainder = include_remainder and remainder > 1
        self._length = self._full_blocks + (1 if self._has_remainder else 0)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self._length:
            raise IndexError(f"Index {index} is out of range for dataset length {self._length}")

        if index < self._full_blocks:
            start = index * self.context_length
            stop = start + self.context_length
            block = np.asarray(self._tokens[start:stop], dtype=np.int64)
        else:
            if not self._has_remainder:
                raise IndexError("Remainder block requested but include_remainder=False")
            if self.pad_token_id is None:
                raise ValueError("pad_token_id is required when include_remainder=True")
            start = self._full_blocks * self.context_length
            block = np.asarray(self._tokens[start:], dtype=np.int64)
            if block.size < self.context_length:
                pad = np.full(self.context_length - block.size, self.pad_token_id, dtype=np.int64)
                block = np.concatenate([block, pad], axis=0)

        x = torch.from_numpy(block[:-1].copy()).long()
        y = torch.from_numpy(block[1:].copy()).long()
        return x, y


def collate_with_padding(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    *,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not batch:
        raise ValueError("Empty batch.")

    max_len = max(x.shape[0] for x, _ in batch)
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for x, y in batch:
        if x.shape[0] < max_len:
            pad_len = max_len - x.shape[0]
            x = torch.cat([x, torch.full((pad_len,), pad_token_id, dtype=x.dtype)], dim=0)
            y = torch.cat([y, torch.full((pad_len,), pad_token_id, dtype=y.dtype)], dim=0)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def make_collate_fn(pad_token_id: int) -> Callable[[List[Tuple[torch.Tensor, torch.Tensor]]], Tuple[torch.Tensor, torch.Tensor]]:
    def _collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        return collate_with_padding(batch, pad_token_id=pad_token_id)

    return _collate


def _require_special_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise RuntimeError(f"Missing special token in tokenizer: {token}")
    return int(token_id)


def create_dataset_bundle(
    *,
    tokenizer_path: Path,
    corpus_dirs: Sequence[Path],
    cache_name: str,
    context_length: int = 1024,
    cache_dir: Path = Path("data/nanogpt/streams"),
    train_fraction: float = 0.9,
    seed: int = 1337,
    max_documents: Optional[int] = None,
    include_remainder: bool = False,
    force_rebuild: bool = False,
) -> DatasetBundle:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_token_id = _require_special_id(tokenizer, SPECIAL_TOKENS["pad"])
    doc_token_id = _require_special_id(tokenizer, SPECIAL_TOKENS["doc"])
    bos_token_id = _require_special_id(tokenizer, SPECIAL_TOKENS["bos"])
    eos_token_id = _require_special_id(tokenizer, SPECIAL_TOKENS["eos"])

    artifacts = prepare_token_streams(
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        corpus_dirs=corpus_dirs,
        cache_dir=cache_dir,
        cache_name=cache_name,
        train_fraction=train_fraction,
        seed=seed,
        max_documents=max_documents,
        force_rebuild=force_rebuild,
        doc_token=SPECIAL_TOKENS["doc"],
    )

    train_dataset = TokenBlockDataset(
        artifacts.train_path,
        context_length=context_length,
        pad_token_id=pad_token_id,
        include_remainder=include_remainder,
    )
    val_dataset = TokenBlockDataset(
        artifacts.val_path,
        context_length=context_length,
        pad_token_id=pad_token_id,
        include_remainder=include_remainder,
    )

    return DatasetBundle(
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        pad_token_id=pad_token_id,
        doc_token_id=doc_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        vocab_size=tokenizer.get_vocab_size(),
        artifacts=artifacts,
    )
