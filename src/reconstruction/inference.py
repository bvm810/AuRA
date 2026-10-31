import os
import torch
import argparse
import torchaudio
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.common.dataset import AudioDataset
from src.common.constants import EVAL_SAMPLE_RATE
from src.reconstruction.config import CONFIGS
from src.reconstruction.system import LitRAEDecoder


def main():
    parser = argparse.ArgumentParser(description="Reconstruction inference script for AuRA decoders.")
    parser.add_argument("--config_id", type=str, required=True, help="Configuration ID from src/reconstruction/config.py")
    parser.add_argument("--checkpoint", type=str, default=None, help="Overrides the config's evaluation checkpoint when set.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for reproducibility.")
    args = parser.parse_args()

    config = CONFIGS[args.config_id]
    set_seeds(args.seed)

    checkpoint = args.checkpoint or config["evaluation"]["checkpoint"]
    results, summary = run_inference(config, args.config_id, checkpoint)
    save_dir = config["evaluation"]["save"]["save_dir"]
    results.to_csv(os.path.join(save_dir, "results.csv"), index=False)
    summary.to_csv(os.path.join(save_dir, "summary.csv"), index=False, float_format="%.3f")

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_system(config_id: str, checkpoint: str) -> LitRAEDecoder:
    print("Loading model")
    system = LitRAEDecoder.load_from_checkpoint(checkpoint, config_id=config_id)
    system.freeze()
    system.ema.copy_to()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return system.to(device)

def build_loader(config: dict) -> DataLoader:
    print("Loading dataset")
    dataset = AudioDataset(deterministic_crop=True, split="test", **config["evaluation"]["dataset"])
    # caption metadata lists a clip once per caption for AudioCaps, but each clip only needs to be reconstructed once
    # for WavCaps there is a single caption per clip, so no need to drop duplicates
    dataset.metadata = dataset.metadata.drop_duplicates(dataset.audio_column, ignore_index=True)
    dataset._refresh_paths()
    return DataLoader(dataset, **config["evaluation"]["dataloaders"])

def create_output_dirs(config: dict) -> dict:
    save = config["evaluation"]["save"]
    os.makedirs(save["save_dir"], exist_ok=True)
    dirs = {}
    for key in ("original", "reconstructed"):
        if save[f"save_{key}_audio"]:
            dirs[key] = os.path.join(save["save_dir"], f"{key}-audio")
            os.makedirs(dirs[key], exist_ok=True)
    return dirs

def run_inference(config: dict, config_id: str, checkpoint: str):
    visqol = config["evaluation"]["save"]["save_audio_in_visqol_format"]
    system = load_system(config_id, checkpoint)
    loader = build_loader(config)
    dirs = create_output_dirs(config)
    sr = loader.dataset.preprocessing_params["sample_rate"]

    print("Reconstructing samples")
    records = []
    for audio, paths in tqdm(loader):
        # eval sets are loaded at the training rate expected by the encoder; reference and reconstruction are compared at EVAL_SAMPLE_RATE
        audio = audio.to(system.device)                                         # (B, T) at the dataset rate
        recon = system.decoder.decode(system.encoder(audio))                    # (B, 1, T') at the decoder rate
        recon = to_eval_rate(recon, system.decoder.sample_rate)                 # (B, 1, T_eval)
        audio = to_eval_rate(audio, sr).unsqueeze(1)                            # (B, 1, T_eval), same layout as recon

        for i, path in enumerate(paths):
            stem = file_stem(path)
            row = {}
            if "original" in dirs:
                row["original"] = save_audio(audio[i], dirs["original"], stem, EVAL_SAMPLE_RATE, visqol)
            else:
                row["original"] = os.path.abspath(loader.dataset.resolve_path(path))
            if "reconstructed" in dirs:
                row["reconstructed"] = save_audio(recon[i], dirs["reconstructed"], stem, EVAL_SAMPLE_RATE, visqol)
            for name, loss in system.val_losses.items():
                row[name] = loss["fn"](recon[i:i + 1], audio[i:i + 1]).item()
            records.append(row)

    results = pd.DataFrame(records)
    return results, build_summary(config_id, results, system.val_losses)

def build_summary(config_id: str, results: pd.DataFrame, losses: dict) -> pd.DataFrame:
    row = {"config": config_id}
    for name in losses:
        row[name] = results[name].mean()
        row[f"{name}-sem"] = results[name].sem()
    return pd.DataFrame([row])

def to_eval_rate(x: torch.Tensor, sr: int) -> torch.Tensor:
    return torchaudio.functional.resample(
        x, sr, EVAL_SAMPLE_RATE,
        rolloff=0.99, resampling_method="sinc_interp_kaiser", lowpass_filter_width=64,
    )

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