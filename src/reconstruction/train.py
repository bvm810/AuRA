import torch
import argparse
import lightning as pl

from lightning.pytorch.loggers import TensorBoardLogger

from src.common.factory import get_strategy
from src.reconstruction.config import CONFIGS
from src.reconstruction.system import LitRAEDecoder


def main():
    parser = argparse.ArgumentParser(description="Training script for AuRA decoders.")
    parser.add_argument("--config_id", type=str, required=True, help="Configuration ID from src/reconstruction/config.py")
    parser.add_argument("--seed", type=int, default=45, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume training from the config's checkpoint")
    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)
    print(f"Starting training for config: {args.config_id} with seed: {args.seed}")
    train(CONFIGS[args.config_id], args.config_id, args.resume)

def train(config: dict, config_id: str, resume: bool):
    print("Loading Lightning Module")
    system = LitRAEDecoder(config_id)
    print("Setting up trainer")
    logger = TensorBoardLogger(**config["training"]["logger"], name=config_id)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    strategy = get_strategy(config["training"]["strategy"])
    trainer = pl.Trainer(**config["training"]["trainer"], logger=logger, strategy=strategy)
    print(f"Training with strategy {strategy}")
    ckpt_path = config["training"]["checkpoint"] if resume else None
    if resume:
        print(f"Resuming training from {ckpt_path}")
    trainer.fit(system, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()