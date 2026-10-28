import torch
import lightning as pl

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.utilities import grad_norm

from src.common.factory import get_ema, get_optimizer, get_lr_scheduler, get_system_dataloader, get_reconstruction_losses
from src.common.encoder import get_encoder

from src.reconstruction.decoder import get_decoder
from src.reconstruction.config import CONFIGS

def grad_total_norm(module, norm_type=2):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([torch.norm(g, norm_type) for g in grads]), norm_type)

class LitRAEDecoder(pl.LightningModule):
    def __init__(self, config_id: str):
        super().__init__()
        self.config_id = config_id
        self.configs = CONFIGS[config_id]
        self.overfit_mode = self.configs["training"]["trainer"]["overfit_batches"] > 0
        self.decoder = get_decoder(self.configs["decoder"])
        self.encoder = get_encoder(self.configs["encoder"])
        self.train_sr = self.configs["training"]["dataset"]["preprocessing_params"]["sample_rate"]
        self.ema = get_ema(self.decoder, self.configs["training"]["ema"])
        self.heavy_log = False
        self.num_heavy_log_batches = self.configs["training"]["additional_logging"].get("num_heavy_log_batches", 16)
        self.val_losses = get_reconstruction_losses(self.configs["training"]["validation_losses"])
        self.save_hyperparameters({"config_id": config_id})

    def _freeze_encoder(self):
        self.encoder.eval()
        self.encoder.requires_grad_(False)

    def training_step(self, batch, batch_idx):
        stft = self.decoder.stft_processor(batch)
        emb = self.encoder(batch, self.train_sr)
        del batch
        loss = self.decoder.schedule.training_loss(self.decoder.model, stft, emb)
        self.log("Training loss", loss, sync_dist=True)
        self.set_heavy_logging()
        return loss

    def validation_step(self, batch, batch_idx):
        stft = self.decoder.stft_processor(batch)
        emb = self.encoder(batch, self.train_sr)
        loss = self.decoder.schedule.validation_loss(self.decoder.model, stft, emb)
        del stft
        if self.heavy_log and batch_idx < self.num_heavy_log_batches:
            self.heavy_logging(emb, batch, batch_idx)
        self.log("Validation loss", loss, sync_dist=True)
        return loss

    def heavy_logging(self, embedding, batch, batch_idx):
        reconstructed = self.decoder.decode(embedding)
        n = min(batch.shape[-1], reconstructed.shape[-1])
        original = batch[:, None, :n]
        reconstructed = reconstructed[..., :n]
        if batch_idx == 0:
            self.register_audio(original[0], label="Original Audio")
            self.register_audio(reconstructed[0], label="Reconstructed Audio")
        total_loss = torch.tensor(0.0, device=self.device)
        for loss_name, loss in self.val_losses.items():
            loss_value = loss["fn"](reconstructed, original)
            self.log(f"Validation {loss_name} loss", loss_value)
            total_loss += loss_value * loss["weight"]
        self.log("Validation total loss", total_loss)
        return total_loss

    def register_audio(self, audio: torch.Tensor, label: str):
        audio = audio.detach().float().cpu().numpy().squeeze()
        self.logger.experiment.add_audio(label, audio, self.global_step, sample_rate=self.decoder.sample_rate)

    def set_heavy_logging(self):
        interval = self.configs["training"]["additional_logging"]["heavy_logging_interval"]
        if interval is not None and self.global_step % interval == 0:
            self.heavy_log = True

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update()

    def on_before_optimizer_step(self, optimizer):
        if self.configs["training"]["additional_logging"]["log_total_grad_norm"]:
            self.log_total_grad_norm()
        if self.configs["training"]["additional_logging"]["log_detailed_grad_norm"]:
            self.log_detailed_grad_norm()

    def log_total_grad_norm(self):
        self.log("Total grad norm for decoder", grad_total_norm(self.decoder.model))

    def log_detailed_grad_norm(self):
        norms = grad_norm(self.decoder.model, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.decoder.model.parameters(), self.configs["training"]["optimizer"])
        scheduler = get_lr_scheduler(optimizer, self.configs["training"]["lr_scheduler"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_start(self):
        if self.configs["encoder"].get("compile", False):
            self.encoder.model = torch.compile(self.encoder.model)
        if self.configs["decoder"].get("compile", False):
            self.decoder.model = torch.compile(self.decoder.model)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def on_save_checkpoint(self, checkpoint: dict):
        sd = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith("decoder.")}
        checkpoint["state_dict"] = self._remove_compile_ckpt_prefix(sd)
        checkpoint["ema_weights"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict):
        ema_state_dict = checkpoint.get("ema_weights", None)
        if ema_state_dict is not None:
            self.ema.load_state_dict(ema_state_dict)

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        state_dict = self._remove_compile_ckpt_prefix(state_dict)
        result = super().load_state_dict(state_dict, strict=False)
        missing = [k for k in result.missing_keys if k.startswith("decoder.")]
        if strict and (missing or result.unexpected_keys):
            raise RuntimeError(
                "Error loading decoder weights.\n"
                f"Missing decoder keys: {missing}\n"
                f"Unexpected keys: {result.unexpected_keys}"
            )
        return result

    def _remove_compile_ckpt_prefix(self, state_dict: dict):
        new_sd = {}
        for k, v in state_dict.items():
            parts = k.split(".")
            parts = [p for p in parts if p != "_orig_mod"]
            new_sd[".".join(parts)] = v
        return new_sd

    def on_fit_start(self):
        self.ema.to(self.device)

    def on_validation_start(self):
        self.ema.store()
        self.ema.copy_to()

    def on_validation_epoch_end(self):
        self.ema.restore()
        if self.heavy_log:
            self.heavy_log = False

    def train_dataloader(self):
        dataset_params = self.configs["training"]["dataset"].copy()
        if self.overfit_mode:
            loader_params = {"batch_size": 1, "num_workers": 0, "pin_memory": False, "persistent_workers": False}
            return get_system_dataloader(dataset_params, loader_params, split=None, deterministic_crop=True)
        else:
            loader_params = self.configs["training"]["dataloaders"]["train"].copy()
            return get_system_dataloader(dataset_params, loader_params, split="train", deterministic_crop=False)

    def val_dataloader(self):
        dataset_params = self.configs["training"]["dataset"].copy()
        if self.overfit_mode:
            loader_params = {"batch_size": 1, "num_workers": 0, "pin_memory": False, "persistent_workers": False}
            return get_system_dataloader(dataset_params, loader_params, split=None, deterministic_crop=True)
        else:
            loader_params = self.configs["training"]["dataloaders"]["validation"].copy()
            return get_system_dataloader(dataset_params, loader_params, split="validation", deterministic_crop=True)
        
    # if needed add new callbacks here
    def configure_callbacks(self):
        ckpt_cfg = self.configs["training"]["checkpointing"]
        ckpt_cfg = ckpt_cfg if isinstance(ckpt_cfg, list) else [ckpt_cfg]
        checkpointers = [ModelCheckpoint(**cfg) for cfg in ckpt_cfg]
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [*checkpointers, lr_monitor]

    # for debugging purposes
    def get_vram(self):
        free = torch.cuda.mem_get_info()[0] / 1024 ** 3
        total = torch.cuda.mem_get_info()[1] / 1024 ** 3
        total_cubes = 24
        free_cubes = int(total_cubes * free / total)
        return f'VRAM: {total - free:.2f}/{total:.2f}GB\t VRAM:[' + (total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'