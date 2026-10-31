"""
Export the EMA weights of a Lightning checkpoint as a plain torch checkpoint.

The config ID selects what is exported:
  - an ID from `src/reconstruction/config.py` loads a `LitRAEDecoder` and exports the DiT decoder (`system.decoder.model`);
  - an ID from `src/generation/config.py` loads a `LitRAEGenerator` and exports the generator DiT (`system.generator.model`).
The Lightning checkpoint is the config's evaluation checkpoint, `config["evaluation"]["checkpoint"]`.

The script copies the EMA weights stored in the checkpoint into the network, checks that this changed the
weights (otherwise the checkpoint held no EMA weights, or they did not load), and saves the network's state
dict with `torch.save`. The exported file loads with `network.load_state_dict(torch.load(out_path))`; for the
DiT decoders this is what `DiTDecoderWrapper` does with the `decoder.model.checkpoint_path` of the generation configs.

    python -m src.utils.export_lightning_ckpt --config_id reconstruction-matpac --out_path checkpoints/reconstruction-matpac/decoder.pt
    python -m src.utils.export_lightning_ckpt --config_id generation-matpac --out_path checkpoints/generation-matpac/generator.pt
"""

import copy
import argparse
import torch

from src.reconstruction.config import CONFIGS as RECONSTRUCTION_CONFIGS
from src.reconstruction.system import LitRAEDecoder
from src.generation.config import CONFIGS as GENERATION_CONFIGS
from src.generation.system import LitRAEGenerator


def main():
    parser = argparse.ArgumentParser(description="Exporting the EMA weights of a DiT decoder or generator checkpoint.")
    parser.add_argument("--config_id", type=str, required=True, help="Configuration ID from src/reconstruction/config.py or src/generation/config.py")
    parser.add_argument("--out_path", type=str, required=True, help="Path to the exported checkpoint file")
    args = parser.parse_args()

    # setup: the config ID tells which Lightning module to load and which network to export
    if args.config_id in RECONSTRUCTION_CONFIGS:
        checkpoint = RECONSTRUCTION_CONFIGS[args.config_id]["evaluation"]["checkpoint"]
        system = LitRAEDecoder.load_from_checkpoint(checkpoint, config_id=args.config_id, map_location="cpu")
        model = system.decoder.model
    else:
        checkpoint = GENERATION_CONFIGS[args.config_id]["evaluation"]["checkpoint"]
        system = LitRAEGenerator.load_from_checkpoint(checkpoint, config_id=args.config_id, map_location="cpu")
        model = system.generator.model
    system.eval()

    # fetch weights
    before = copy.deepcopy(model.state_dict())
    system.ema.copy_to()
    after = model.state_dict()

    # check that EMA are not the same as the non-EMA weights
    delta = max((after[k].float() - before[k].float()).abs().max().item() for k in before)
    n_changed = sum((after[k] != before[k]).any().item() for k in before)
    print(f"{n_changed}/{len(before)} tensors changed, max delta {delta:.3e}")
    assert delta > 0, "copy_to was a no-op — ckpt already held EMA weights, or ema didn't load"

    # save weights
    final = {k: v.detach().cpu().clone() for k, v in after.items()}
    torch.save(final, args.out_path)
    print(f"Exported {checkpoint} to {args.out_path}")

if __name__ == "__main__":
    main()