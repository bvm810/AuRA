import os
import torch
import argparse
import torchaudio
import numpy as np
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader

from src.common.dataset import CaptioningDataset
from src.generation.config import CONFIGS
from src.generation.system import LitRAEGenerator, frechet_distance


def main():
    parser = argparse.ArgumentParser(description="Inference script for text-to-audio generation.")
    parser.add_argument("--config_id", type=str, required=True, help="Configuration ID from src/generation/config.py")
    parser.add_argument("--checkpoint", type=str, default=None, help="Overrides the config's evaluation checkpoint when set.")
    parser.add_argument("--split", type=str, default="test", help="Split of the evaluation dataset, e.g. 'validation' for CFG selection.")
    parser.add_argument("--cfg_scale", type=float, default=None, help="Overrides the config's generator cfg_scale when set.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for reproducibility.")
    args = parser.parse_args()

    config = CONFIGS[args.config_id]
    set_seeds(args.seed)

    # Namespace outputs by the cfg_scale actually used, so a CFG sweep over one checkpoint doesn't overwrite itself.
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else config["generator"]["cfg_scale"]
    save_dir = os.path.join(config["evaluation"]["save"]["save_dir"], f"cfg-scale-{cfg_scale}")

    checkpoint = args.checkpoint or config["evaluation"]["checkpoint"]
    results, summary = run_inference(config, args.config_id, checkpoint, args.split, cfg_scale, save_dir)
    results.to_csv(os.path.join(save_dir, "results.csv"), index=False)
    summary.to_csv(os.path.join(save_dir, "summary.csv"), index=False, float_format="%.3f")

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clap_similarity(audio_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    return (F.normalize(audio_emb, dim=-1) * F.normalize(text_emb, dim=-1)).sum(dim=-1)

def load_system(config_id: str, checkpoint: str, cfg_scale: float) -> LitRAEGenerator:
    print("Loading model")
    system = LitRAEGenerator.load_from_checkpoint(checkpoint, config_id=config_id)
    system.freeze()
    system.ema.copy_to()
    system.generator.cfg_scale = cfg_scale  # Generator.sample reads it at call time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return system.to(device)

def build_loader(config: dict, split: str) -> DataLoader:
    print("Loading dataset")
    dataset = CaptioningDataset(deterministic_crop=True, split=split, **config["evaluation"]["dataset"])
    return DataLoader(dataset, **config["evaluation"]["dataloaders"])

def run_inference(config: dict, config_id: str, checkpoint: str, split: str, cfg_scale: float, save_dir: str):
    visqol = config["evaluation"]["save"]["save_audio_in_visqol_format"]
    system = load_system(config_id, checkpoint, cfg_scale)
    loader = build_loader(config, split)
    out_dir = os.path.join(save_dir, "generated-audio")
    os.makedirs(out_dir, exist_ok=True)
    sr, eval_sr = loader.dataset.preprocessing_params["sample_rate"], system.eval_sample_rate

    print("Generating samples")
    records = []
    seen = Counter()
    real_fad, fake_fad = [], []   # CLAP audio embeddings accumulated for FAD

    for audio, captions, paths in tqdm(loader):
        # everything is brought to the eval sample rate before scoring / saving
        real = system._to_eval_rate(audio.to(system.device), sr)                # (B, T_eval), FAD reference
        condition, condition_mask = system.text_encoder(captions)
        fake_latent = system.generator.sample(condition, condition_mask)
        fake = system._to_eval_rate(system.decoder.decode(fake_latent), system.decoder.sample_rate)    # (B, 1, T_eval)
        n = min(real.shape[-1], fake.shape[-1])  # decoders at other sample rates can be a few samples longer

        real_fad.append(system.clap.embed_audio(real[..., :n], sr=eval_sr))
        fake_emb = system.clap.embed_audio(fake[..., :n], sr=eval_sr)
        fake_fad.append(fake_emb)
        clap = clap_similarity(fake_emb, system.clap.embed_text(captions))

        for i, (caption, path) in enumerate(zip(captions, paths)):
            stem = file_stem(path)
            seen[stem] += 1
            if seen[stem] > 1:  # clips with several captions (AudioCaps) get one output per caption
                stem = f"{stem}-{seen[stem]}-caption"
            records.append({
                "caption": caption,
                "test-file": os.path.abspath(loader.dataset.resolve_path(path)),
                "generated": save_audio(fake[i, :, :n], out_dir, stem, eval_sr, visqol),
                "clap-audio-text": clap[i].item(),
            })

    results = pd.DataFrame(records)
    summary = pd.DataFrame([{
        "config": config_id,
        "fad-audio": frechet_distance(torch.cat(real_fad), torch.cat(fake_fad)).item(),
        "clap-audio": results["clap-audio-text"].mean(),
        "clap-audio-sem": results["clap-audio-text"].sem(),
    }])
    return results, summary

def file_stem(rel_path: str) -> str:
    # parent-dir prefix disambiguates basename collisions across sources,
    # e.g. ".../soundbible/12345.wav" -> "soundbible-12345"
    parent = os.path.basename(os.path.dirname(rel_path))
    name = os.path.splitext(os.path.basename(rel_path))[0]
    return f"{parent}-{name}" if parent else name

def save_audio(audio: torch.Tensor, out_dir: str, stem: str, sample_rate: int, visqol: bool) -> str:
    audio = audio.cpu()  # (1, T)
    if visqol:  # ViSQOL wants 48 kHz 16-bit PCM
        path = os.path.join(out_dir, f"{stem}.wav")
        audio = torchaudio.functional.resample(
            audio, sample_rate, 48000,
            rolloff=0.99, resampling_method="sinc_interp_kaiser", lowpass_filter_width=64,
        )
        torchaudio.save(path, audio, 48000, encoding="PCM_S", bits_per_sample=16)
    else:
        path = os.path.join(out_dir, f"{stem}.flac")
        torchaudio.save(path, audio, sample_rate)
    return os.path.abspath(path)


if __name__ == "__main__":
    main()