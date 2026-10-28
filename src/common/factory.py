import torch
import auraloss

from torch.optim import RAdam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LinearLR, SequentialLR, ConstantLR, ExponentialLR
from torch_ema import ExponentialMovingAverage
from torch.utils.data import DataLoader
from lightning.pytorch.strategies import DDPStrategy

from src.common.dataset import AudioDataset, CaptioningDataset

def get_ema(model: torch.nn.Module, ema_config: dict):
    return ExponentialMovingAverage(model.parameters(), **ema_config)

def get_optimizer(parameters: list, optimizer_config: dict):
    if optimizer_config["type"] == "RAdam":
        return RAdam(parameters, **optimizer_config["params"])
    if optimizer_config["type"] == "AdamW":
        return AdamW(parameters, **optimizer_config["params"])
    raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")

def get_lr_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: dict):
    if scheduler_config["type"] == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, **scheduler_config["params"])
    if scheduler_config["type"] == "OneCycleLR":
        return OneCycleLR(optimizer, **scheduler_config["params"])
    if scheduler_config["type"] == "RampUpCosineAnnealingLR":
        params = scheduler_config["params"].copy()
        warmup_iters = params.pop("warmup_iters")
        slope = 1 / warmup_iters
        warmup = LinearLR(optimizer, start_factor=slope, total_iters=warmup_iters)
        scheduler = CosineAnnealingLR(optimizer, **params)
        return SequentialLR(optimizer, [warmup, scheduler], milestones=[warmup_iters])
    if scheduler_config["type"] == "ConstantLR":
        return ConstantLR(optimizer, factor=scheduler_config["params"]["factor"], total_iters=scheduler_config["params"]["total_iters"])
    if scheduler_config["type"] == "ExponentialLR":
        return ExponentialLR(optimizer, **scheduler_config["params"])
    raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")

def get_strategy(strategy_config: dict):
    if strategy_config["type"] == "DDP":
        return DDPStrategy(**strategy_config["params"])
    return "auto"

def get_system_dataloader(dataset_params: dict, dataloader_params: dict, split: str = None, deterministic_crop: bool = False) -> DataLoader:
    if "caption_column" in dataset_params:
        dataset = CaptioningDataset(split=split, deterministic_crop=deterministic_crop, **dataset_params)
    else:
        dataset = AudioDataset(split=split, deterministic_crop=deterministic_crop, **dataset_params)
    loader = DataLoader(dataset, **dataloader_params)
    return loader
    

def get_reconstruction_losses(losses_config: dict) -> dict:
    losses = {}
    for loss_name, loss_config in losses_config.items():
        params = loss_config.get("params", {})
        weight = loss_config.get("weight", 1.0)
        losses[loss_name] = {
            "fn": get_loss(loss_name, params),
            "weight": weight,
        }
    return losses

def get_loss(loss_name: str, params: dict = {}):
    # mel-spec losses
    if loss_name == "L2":
        return torch.nn.MSELoss(**params)
    if loss_name == "L1":
        return torch.nn.L1Loss(**params)

    # waveform losses
    if loss_name == "MR-STFT":
        return auraloss.freq.MultiResolutionSTFTLoss(**params)
    if loss_name == "SI-SDR":
        return auraloss.time.SISDRLoss(**params)
    if loss_name == "SC":
        return auraloss.freq.SpectralConvergenceLoss(**params)

    raise ValueError(f"Unsupported loss type: {loss_name}")