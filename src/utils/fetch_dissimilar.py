"""
Find the test captions most underrepresented in the training set, i.e. the CLAP100 subsets of the paper.

How it works:
    Every caption is embedded with the generators' text encoder (`T5EncoderWrapper`, T5-base), mean-pooled over
    its non-padding tokens and L2-normalised. Each test caption is scored by the mean cosine distance to its
    `--k` nearest training captions (exact search with a faiss inner-product index). Test captions are then
    taken in order of decreasing score, keeping at most one caption per audio file, until `--n` are selected.

Inputs:
    --train_csv   training metadata with `split` and `caption` columns; only the rows with split == "train" are
                  used, e.g. the generation training CSV (`long-segments.csv` in WAVCAPS_DIR).
    --test_csv    test set with `test-file` and `caption` columns, e.g. the `results.csv` written by
                  `src/generation/inference.py` on AudioCaps-test or WavCaps-test.

Output:
    `--out_path` (default `100-dissimilar.csv`), one row per selected sample, sorted by decreasing distance, with
    the columns `caption`, `test-file` and `mean-distance`. The selection depends only on the captions, so it is
    computed once per test set and shared by all models. CLAP100 is the mean CLAP score of a model's inference
    results over these rows:

        results = pd.read_csv("results/generation-matpac/audiocaps/cfg-scale-5.0/results.csv")
        clap100 = results.merge(pd.read_csv("100-dissimilar.csv"), on=["caption", "test-file"])["clap-audio-text"].mean()

Usage:
    python -m src.utils.fetch_dissimilar \
        --train_csv  ~/data/captioning/long-segments.csv \
        --test_csv   results/generation-matpac/audiocaps/cfg-scale-5.0/results.csv \
        --out_path   results/100-dissimilar-audiocaps.csv
"""

import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.common.constants import TEXT_ENCODER_DEFAULT_CONFIG
from src.common.encoder import get_encoder


# --------------------------------------------------------------------------- #
# Settings                                                                     #
# --------------------------------------------------------------------------- #

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# T5-base as in the generators, tokenized as when the paper's CLAP100 subsets were selected
T5_CONFIG = {
    **TEXT_ENCODER_DEFAULT_CONFIG,
    "tokenization_params": {"padding": "longest", "truncation": True, "max_length": 128},
}
BATCH_SIZE = 256
K_NEIGHBORS = 10  # neighbours in the train set used to score each test caption
N_OUTPUT = 100    # number of test samples to keep

SPLIT_COLUMN = "split"
CAPTION_COLUMN = "caption"
TEST_FILE_COLUMN = "test-file"
DISTANCE_COLUMN = "mean-distance"

DEFAULT_OUTPUT = "100-dissimilar.csv"


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #

def load_train_captions(path: Path) -> list[str]:
    df = pd.read_csv(path)
    df = df[df[SPLIT_COLUMN] == "train"]
    if df.empty:
        raise ValueError("No train captions found in the dataset.")
    return df[CAPTION_COLUMN].tolist()

def load_test_set(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[TEST_FILE_COLUMN, CAPTION_COLUMN]]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Encoding                                                                     #
# --------------------------------------------------------------------------- #

def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the token axis, ignoring padding. (B, L, D) -> (B, D)."""
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1)
    return summed / counts

@torch.no_grad()
def encode_captions(model: torch.nn.Module, captions: list[str], batch_size: int) -> np.ndarray:
    """Encode captions into L2-normalised float32 vectors of shape (N, D)."""
    chunks = []
    for start in tqdm(range(0, len(captions), batch_size)):
        batch = captions[start : start + batch_size]
        hidden, attention_mask = model.encode(batch)
        pooled = masked_mean_pool(hidden, attention_mask)
        pooled = F.normalize(pooled.float(), p=2, dim=-1)
        chunks.append(pooled.cpu().numpy())
    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    return np.ascontiguousarray(embeddings)


# --------------------------------------------------------------------------- #
# kNN                                                                          #
# --------------------------------------------------------------------------- #

def build_index(train_embeddings: np.ndarray):
    dim = train_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) # exact inner product index --> cosine similarity since inputs are unit-norm
    index.add(train_embeddings)
    return index

def mean_neighbour_distance(index, test_embeddings: np.ndarray, k: int) -> np.ndarray:
    similarities, _ = index.search(test_embeddings, k)
    return (1.0 - similarities).mean(axis=1)


# --------------------------------------------------------------------------- #
# Selection                                                                    #
# --------------------------------------------------------------------------- #

def select_top_n(test_df: pd.DataFrame, distances: np.ndarray, n: int) -> pd.DataFrame:
    """Take the n most dissimilar captions, at most one per audio file.

    Captions are walked in order of decreasing distance; a caption is skipped if
    its audio file has already been selected, so the list keeps filling with the
    next most dissimilar unseen file until it reaches n.
    """
    order = np.argsort(-distances, kind="stable")

    seen = set()
    selected = []
    for idx in order:
        test_file = test_df.at[idx, TEST_FILE_COLUMN]
        if test_file in seen:
            continue
        seen.add(test_file)
        selected.append(int(idx))
        if len(selected) == n:
            break

    if len(selected) < n:
        print(f"[warn] only {len(selected)} distinct audio file(s) available, requested {n}")

    out = test_df.loc[selected, [CAPTION_COLUMN, TEST_FILE_COLUMN]].copy()
    out[DISTANCE_COLUMN] = distances[selected]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the test samples whose captions are most dissimilar to the training set.")
    parser.add_argument("--train_csv", type=Path, required=True, help=f"Training metadata CSV with '{SPLIT_COLUMN}' and '{CAPTION_COLUMN}' columns.")
    parser.add_argument("--test_csv", type=Path, required=True, help=f"Test set CSV with '{TEST_FILE_COLUMN}' and '{CAPTION_COLUMN}' columns.")
    parser.add_argument("--out_path", type=Path, default=Path(DEFAULT_OUTPUT), help=f"Output CSV path (default: '{DEFAULT_OUTPUT}').")
    parser.add_argument("--k", type=int, default=K_NEIGHBORS, help=f"Number of nearest neighbours (default: {K_NEIGHBORS}).")
    parser.add_argument("--n", type=int, default=N_OUTPUT, help=f"Number of samples to output (default: {N_OUTPUT}).")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help=f"Encoding batch size (default: {BATCH_SIZE}).")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    train_captions = load_train_captions(args.train_csv)
    test_df = load_test_set(args.test_csv)
    print(f"{len(train_captions)} train caption(s), {len(test_df)} test caption(s) over {test_df[TEST_FILE_COLUMN].nunique()} audio file(s)")
    print(f"Loading T5 encoder on {DEVICE}...")
    model = get_encoder(T5_CONFIG).to(DEVICE).eval()
    print("Encoding train captions...")
    train_embeddings = encode_captions(model, train_captions, args.batch_size)
    print("Encoding test captions...")
    test_embeddings = encode_captions(model, test_df[CAPTION_COLUMN].tolist(), args.batch_size)
    print(f"Building index over {train_embeddings.shape[0]} vectors...")
    index = build_index(train_embeddings)
    print(f"Searching {args.k} nearest neighbours...")
    distances = mean_neighbour_distance(index, test_embeddings, args.k)
    result = select_top_n(test_df, distances, args.n)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_path, index=False)
    print(f"Wrote {len(result)} row(s) to {args.out_path} (mean-distance range: {result[DISTANCE_COLUMN].min():.4f} - {result[DISTANCE_COLUMN].max():.4f})")


if __name__ == "__main__":
    main()