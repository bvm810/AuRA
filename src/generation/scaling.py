import os
import re
import glob
import torch
import argparse
import pandas as pd

from tqdm import tqdm

from src.generation.config import CONFIGS
from src.generation.system import LitRAEGenerator, frechet_distance
from src.generation.inference import set_seeds, clap_similarity, build_loader


SCALING_COLS = ["checkpoint", "steps", "clap-score", "fad", "flops", "flops-estimated"]
CKPT_GLOB = "epoch=*-step=*.ckpt"


def main():
    parser = argparse.ArgumentParser(description="Scaling-curve evaluation over the checkpoints of a training run.")
    parser.add_argument("--config_id", type=str, required=True, help="Configuration ID from src/generation/config.py")
    parser.add_argument("--split", type=str, default="test", help="Split of the evaluation dataset, e.g. 'validation'.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for reproducibility; re-applied before every checkpoint.")
    parser.add_argument("--cfg_scale", type=float, default=None, help="Overrides the config's generator cfg_scale when set.")
    parser.add_argument("--max_checkpoints", type=int, default=None, help="Evaluate at most this many checkpoints (debugging).")
    args = parser.parse_args()
    config = CONFIGS[args.config_id]
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else config["generator"]["cfg_scale"]
    # Same namespacing as inference.py, plus a scaling/ subdir so the curve does not collide with the single-checkpoint outputs of that run.
    save_dir = os.path.join(config["evaluation"]["save"]["save_dir"], f"cfg-scale-{cfg_scale}", "scaling")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "scaling-results.csv")
    checkpoints = find_checkpoints(config, csv_path, args.max_checkpoints)
    print(f"Evaluating {len(checkpoints)} checkpoints, appending to {csv_path}")
    run_scaling(config, args.config_id, checkpoints, csv_path, args.split, args.seed, cfg_scale)

def step_of(path: str) -> int:
    return int(re.search(r"step=(\d+)", os.path.basename(path)).group(1))

def find_checkpoints(config: dict, csv_path: str, max_checkpoints: int = None) -> list:
    # The checkpoint dir is the one holding the config's training checkpoint (last.ckpt).
    ckpt_dir = os.path.dirname(config["training"]["checkpoint"])
    paths = sorted(glob.glob(os.path.join(ckpt_dir, CKPT_GLOB)), key=step_of)
    if os.path.exists(csv_path):   # resume: filenames already in the csv are skipped unopened
        done = set(pd.read_csv(csv_path)["checkpoint"])
        paths = [p for p in paths if os.path.basename(p) not in done]
    return paths[:max_checkpoints]

def load_system(config_id: str, cfg_scale: float) -> LitRAEGenerator:
    # Built from the config rather than from a checkpoint: the generator weights are
    # overwritten per checkpoint anyway, and the frozen modules are loaded only once.
    print("Loading model")
    system = LitRAEGenerator(config_id)
    system.freeze()
    system.generator.cfg_scale = cfg_scale  # Generator.sample reads it at call time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return system.to(device)

def load_checkpoint(system: LitRAEGenerator, path: str) -> int:
    # In-place weight swap; torch_ema holds live refs to generator.model parameters,
    # so the restored EMA weights are copied straight into them, as in inference.py.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    system.load_state_dict(ckpt["state_dict"])
    system.on_load_checkpoint(ckpt)   # restores ema_weights and train_flops
    system.ema.copy_to()
    return ckpt["global_step"]

def global_batch_size(config: dict) -> int:
    trainer = config["training"]["trainer"]
    batch_size = config["training"]["dataloaders"]["train"]["batch_size"]
    return batch_size * trainer["devices"] * trainer["num_nodes"] * trainer.get("accumulate_grad_batches", 1)

def run_scaling(config: dict, config_id: str, checkpoints: list, csv_path: str, split: str, seed: int, cfg_scale: float):
    system = load_system(config_id, cfg_scale)
    loader = build_loader(config, split)
    batch = global_batch_size(config)
    cache = {}   # real / text CLAP embeddings, constant across checkpoints
    for path in checkpoints:
        print(f"Evaluating {os.path.basename(path)}")
        set_seeds(seed)   # same noise at every checkpoint
        steps = load_checkpoint(system, path)
        clap, fad = evaluate(system, loader, cache)
        row = {
            "checkpoint": os.path.basename(path),
            "steps": steps,
            "clap-score": clap,
            "fad": fad,
            "flops": system.train_flops,
            "flops-estimated": system.flops_per_sample * batch * steps,
        }
        print(f"steps {steps}: clap {clap:.4f}, fad {fad:.3f}, {row['flops'] / 1e15:.1f} PFLOPs")
        pd.DataFrame([row])[SCALING_COLS].to_csv(
            csv_path,
            mode="a",
            header=not os.path.exists(csv_path),
            index=False,
        )

def evaluate(system: LitRAEGenerator, loader, cache: dict) -> tuple:
    sr, eval_sr = loader.dataset.preprocessing_params["sample_rate"], system.eval_sample_rate
    sims, fake_fad = [], []

    for i, (audio, captions, _) in enumerate(tqdm(loader)):
        condition, condition_mask = system.text_encoder(captions)
        fake_latent = system.generator.sample(condition, condition_mask)
        fake = system._to_eval_rate(system.decoder.decode(fake_latent), system.decoder.sample_rate)

        # The deterministic crop makes the real arm identical on every pass, so the FAD
        # reference and the caption embeddings are computed for the first checkpoint only.
        if i not in cache:
            real = system._to_eval_rate(audio.to(system.device), sr)
            n = min(real.shape[-1], fake.shape[-1])
            cache[i] = {
                "real": system.clap.embed_audio(real[..., :n], sr=eval_sr),
                "text": system.clap.embed_text(captions),
                "length": n,
            }
        cached = cache[i]

        fake_emb = system.clap.embed_audio(fake[..., :cached["length"]], sr=eval_sr)
        sims.append(clap_similarity(fake_emb, cached["text"]))
        fake_fad.append(fake_emb)

    real_emb = torch.cat([cache[i]["real"] for i in sorted(cache)])
    fad = frechet_distance(real_emb, torch.cat(fake_fad))
    return torch.cat(sims).mean().item(), fad.item()


if __name__ == "__main__":
    main()