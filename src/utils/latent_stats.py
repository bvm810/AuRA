#!/usr/bin/env python3
"""
Compute the scalar latent mean / std that `Generator.normalize` needs.

Takes a `config_id` from `src/generation/config.py`, rebuilds *that config's* audio encoder
through `get_encoder`, runs it over the train split of *that config's* dataset, and accumulates
the global (all-element) mean and std of the latents -- the SD convention scalars that go into
`generator.latent_mean` / `generator.latent_std`.

Results are printed and written to `tmp.txt` in the current working directory.

    python -m src.utils.latent_stats --config_id generation-matpac
    python -m src.utils.latent_stats --config_id generation-beats --max_batches -1
"""

import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import lightning as pl

from src.common.encoder import get_encoder
from src.common.factory import get_system_dataloader
from src.generation.config import CONFIGS


SD_BASE_DIM = 4096  # 32 * 32 * 4, the SD3 base dimension the shift is measured against


############################### Accumulators ###############################


class ScalarMoments:
    """
    Global mean/std over every element of every latent, accumulated batch by batch
    as running sums: std = sqrt(E[x^2] - E[x]^2). Float64 throughout.

    This form loses precision when |mean| >> std, since the variance is then the
    difference of two nearly equal large numbers. Pooled over all dimensions the
    latents sit near zero mean, so that is not the operating regime here -- but a
    variance that comes out negative is the tell, and `std` warns if it does.
    """

    def __init__(self, device: torch.device):
        self.count = 0
        self.total = torch.zeros((), dtype=torch.float64, device=device)
        self.total_sq = torch.zeros((), dtype=torch.float64, device=device)
        self.minimum = torch.full((), float("inf"), dtype=torch.float64, device=device)
        self.maximum = torch.full((), float("-inf"), dtype=torch.float64, device=device)

    def update(self, x: torch.Tensor) -> None:
        x = x.double()
        if x.numel() == 0:
            return
        self.count += x.numel()
        self.total += x.sum()
        self.total_sq += x.pow(2).sum()
        self.minimum = torch.minimum(self.minimum, x.min())
        self.maximum = torch.maximum(self.maximum, x.max())

    @property
    def mean(self) -> torch.Tensor:
        return self.total / max(self.count, 1)

    @property
    def std(self) -> torch.Tensor:
        # population std; at count ~1e9 the unbiased correction is ~1e-9 relative
        var = self.total_sq / max(self.count, 1) - self.mean.pow(2)
        if var < 0:
            print(f"WARNING: variance came out negative ({float(var):.3e}) -- E[x^2] and E[x]^2 "
                  "have cancelled, the reported std is not trustworthy")
        return var.clamp(min=0).sqrt()


class PerDimMoments:
    """
    Per-dimension mean/std. Not used for normalization -- the generator normalizes with
    scalars -- but a wide spread across dimensions is the signal that the scalar
    convention is papering over a badly conditioned latent space, so it is worth seeing.
    """

    def __init__(self, latent_dim: int, device: torch.device):
        self.count = 0
        self.total = torch.zeros(latent_dim, dtype=torch.float64, device=device)
        self.total_sq = torch.zeros(latent_dim, dtype=torch.float64, device=device)

    def update(self, x: torch.Tensor) -> None:
        x = x.double().flatten(0, -2)  # (B, N, D) -> (B * N, D)
        self.count += x.shape[0]
        self.total += x.sum(0)
        self.total_sq += x.pow(2).sum(0)

    @property
    def mean(self) -> torch.Tensor:
        return self.total / max(self.count, 1)

    @property
    def std(self) -> torch.Tensor:
        var = self.total_sq / max(self.count, 1) - self.mean.pow(2)
        return var.clamp(min=0).sqrt()


################################## Helpers #################################


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)


def build_loader(config: dict, args: argparse.Namespace):
    dataset_params = dict(config["training"]["dataset"])
    loader_params = dict(config["training"]["dataloaders"]["train"])
    # Shuffle: with a batch cap an unshuffled loader would only ever see the head of the
    # csv, i.e. possibly a single source corpus. Keep the tail too.
    loader_params["shuffle"] = True
    loader_params["drop_last"] = False
    if args.batch_size is not None:
        loader_params["batch_size"] = args.batch_size
    if args.num_workers is not None:
        loader_params["num_workers"] = args.num_workers
        loader_params["persistent_workers"] = args.num_workers > 0

    return get_system_dataloader(
        dataset_params,
        loader_params,
        split=args.split,
        deterministic_crop=args.deterministic_crop,
    )


def check_latent_shape(config: dict, latent: torch.Tensor) -> tuple[int, int]:
    seq_len, latent_dim = int(latent.shape[1]), int(latent.shape[2])
    declared = config["generator"]["model"]["latent_dim"]
    assert latent_dim == declared, (
        f"{config['audio_encoder']['type']} emits {latent_dim}-dim latents, "
        f"but generator.model.latent_dim is {declared}"
    )
    max_seq_len = config["generator"]["model"]["max_seq_len"]
    assert max_seq_len >= seq_len, f"generator max_seq_len {max_seq_len} < seq_len {seq_len}"
    return seq_len, latent_dim


################################### Main ###################################


def main():
    parser = argparse.ArgumentParser(description="Latent mean/std for the RAE generator.")
    parser.add_argument("--config_id", type=str, required=True, help="Configuration ID from src/generation/config.py")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to measure")
    parser.add_argument("--max_batches", type=int, default=200,
                        help="Batches to accumulate; -1 for the whole split")
    parser.add_argument("--batch_size", type=int, default=None, help="Override the config batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="Override the config worker count")
    parser.add_argument("--deterministic_crop", action="store_true",
                        help="Crop from sample 0 instead of a random offset (training uses random)")
    parser.add_argument("--report_every", type=int, default=10, help="Print the running estimate every N batches")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="tmp.txt", help="Output file, relative to the cwd")
    parser.add_argument("--seed", type=int, default=45, help="Random seed")
    args = parser.parse_args()

    set_seeds(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    config = CONFIGS[args.config_id]

    print(f"Using device: {device}")
    print(f"Config '{args.config_id}': encoder '{config['audio_encoder']['type']}' on split '{args.split}'")
    encoder = get_encoder(config["audio_encoder"]).to(device).eval()
    loader = build_loader(config, args)

    total_batches = len(loader)
    num_batches = total_batches if args.max_batches < 0 else min(args.max_batches, total_batches)
    print(f"{len(loader.dataset)} rows in the split, {total_batches} batches; accumulating over {num_batches}")

    scalar = ScalarMoments(device)
    per_dim = None
    shape = None
    num_segments = 0
    num_nonfinite = 0
    interrupted = False

    try:
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= num_batches:
                break
            audio = batch[0].to(device, non_blocking=True)   # (B, segment_length)
            latent = encoder(audio)                          # (B, seq_len, latent_dim)

            if shape is None:
                shape = check_latent_shape(config, latent)
                per_dim = PerDimMoments(shape[1], device)
                print(f"Latent shape per segment: {shape} (seq_len, latent_dim)")
            assert tuple(latent.shape[1:]) == shape, f"latent shape changed: {tuple(latent.shape[1:])} vs {shape}"

            # a single NaN silently poisons every downstream statistic
            if not torch.isfinite(latent).all():
                num_nonfinite += 1
                print(f"  batch {batch_idx}: non-finite latents, skipped")
                continue

            scalar.update(latent)
            per_dim.update(latent)
            num_segments += latent.shape[0]

            if args.report_every and (batch_idx + 1) % args.report_every == 0:
                print(f"  batch {batch_idx + 1}/{num_batches}  "
                      f"mean {float(scalar.mean):+.6g}  std {float(scalar.std):.6g}  "
                      f"({num_segments} segments)", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted -- reporting the statistics accumulated so far.")

    assert num_segments > 0, "no segments were encoded, nothing to report"

    report = format_report(args.config_id, config, args, scalar, per_dim, shape, num_segments,
                           total_batches, len(loader.dataset), num_nonfinite, interrupted)
    print("\n" + report)
    out_path = Path.cwd() / args.out
    out_path.write_text(report)
    print(f"Written to {out_path}")


def format_report(config_id, config, args, scalar, per_dim, shape, num_segments, total_batches,
                  num_rows, num_nonfinite, interrupted) -> str:
    seq_len, latent_dim = shape
    preprocessing = config["training"]["dataset"]["preprocessing_params"]
    mean, std = float(scalar.mean), float(scalar.std)
    tokens = seq_len * latent_dim
    dim_mean, dim_std = per_dim.mean.cpu(), per_dim.std.cpu()
    quantiles = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)

    def triple(x):
        lo, mid, hi = torch.quantile(x, quantiles).tolist()
        return f"{lo:+.4g} / {mid:+.4g} / {hi:+.4g}"

    lines = [
        f"# Latent statistics for config '{config_id}'",
        f"# generated {datetime.datetime.now().isoformat(timespec='seconds')} by latent_stats.py",
        "#",
        f"# audio encoder     : {config['audio_encoder']['type']}",
        f"# decoder           : {config['decoder']['type']}",
        f"# dataset root      : {config['training']['dataset']['root_dir']}",
        f"# split             : {args.split}  ({num_rows} rows, {total_batches} batches)",
        f"# segment length    : {preprocessing['segment_length']} samples @ {preprocessing['sample_rate']} Hz",
        f"# crop              : {'deterministic (offset 0)' if args.deterministic_crop else 'random'}, seed {args.seed}",
        f"# latent shape      : ({seq_len}, {latent_dim})  (seq_len, latent_dim)",
        f"# segments seen     : {num_segments}",
        f"# elements seen     : {scalar.count:,}",
        "",
        "# --- the two numbers, for generator.latent_mean / generator.latent_std ---",
        f'    "latent_mean": {mean:.6g},',
        f'    "latent_std": {std:.6g},',
        "",
        "# --- diagnostics ---",
        f"# min / max                       : {float(scalar.minimum):+.4g} / {float(scalar.maximum):+.4g}",
        f"# per-dim mean  (min/median/max)  : {triple(dim_mean)}",
        f"# per-dim std   (min/median/max)  : {triple(dim_std)}",
        f"# m = seq_len * latent_dim        : {tokens}",
        f"# alpha = sqrt(m / {SD_BASE_DIM})           : {(tokens / SD_BASE_DIM) ** 0.5:.4g}",
    ]
    if num_nonfinite:
        lines.append(f"# NOTE: {num_nonfinite} batch(es) skipped for non-finite latents")
    if interrupted:
        lines.append("# NOTE: run was interrupted; these are partial statistics")
    if args.max_batches >= 0 and args.max_batches < total_batches:
        lines.append(f"# NOTE: capped at {args.max_batches} of {total_batches} batches "
                     f"(--max_batches -1 for a full pass)")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()