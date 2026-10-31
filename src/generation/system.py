import json
import torch
import torchaudio
import lightning as pl
import torch.nn.functional as F

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.utilities import grad_norm
from lightning.fabric.utilities.throughput import measure_flops

from src.generation.config import CONFIGS
from src.generation.generator import Generator, GeneratorDiT
from src.common.constants import EVAL_SAMPLE_RATE
from src.common.encoder import get_encoder
from src.reconstruction.decoder import get_decoder
from src.common.factory import get_ema, get_optimizer, get_lr_scheduler, get_system_dataloader


def grad_total_norm(module, norm_type=2):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([torch.norm(g, norm_type) for g in grads]), norm_type)

def frechet_distance(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    real, fake = real.float(), fake.float()
    mu_r, mu_f = real.mean(0), fake.mean(0)
    cov_r, cov_f = torch.cov(real.T), torch.cov(fake.T)
    diff = mu_r - mu_f
    eigvals_r, eigvecs_r = torch.linalg.eigh(cov_r)
    sqrt_cov_r = eigvecs_r @ torch.diag(eigvals_r.clamp(min=0).sqrt()) @ eigvecs_r.T
    m_eigvals = torch.linalg.eigvalsh(sqrt_cov_r @ cov_f @ sqrt_cov_r).clamp(min=0)
    return diff.dot(diff) + torch.trace(cov_r) + torch.trace(cov_f) - 2 * m_eigvals.sqrt().sum()

def clap_score(audio_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    return (F.normalize(audio_emb, dim=-1) * F.normalize(text_emb, dim=-1)).sum(dim=-1).mean()

def measure_generator_flops(generator_config: dict, context_len: int) -> int:
    seq_len, latent_dim = generator_config["schedule"]["output_shape"]
    with torch.device("meta"):
        model = GeneratorDiT(**generator_config["model"])
        x = torch.empty(1, seq_len, latent_dim)
        t = torch.empty(1)
        context = torch.empty(1, context_len, generator_config["model"]["context_dim"])
        context_mask = torch.ones(1, context_len, dtype=torch.bool)
    forward = lambda: model(x, t, context, context_mask=context_mask)
    return measure_flops(model, forward, torch.Tensor.sum)


class LitRAEGenerator(pl.LightningModule):
    def __init__(self, config_id: str):
        super().__init__()
        self.config_id = config_id
        self.configs = CONFIGS[config_id]
        self.overfit_mode = self.configs["training"]["trainer"].get("overfit_batches", 0) > 0
        self.dataset_sample_rate = self.configs["training"]["dataset"]["preprocessing_params"]["sample_rate"]
        self.eval_sample_rate = EVAL_SAMPLE_RATE

        # auxiliary modules
        self.audio_encoder = get_encoder(self.configs["audio_encoder"])
        self.text_encoder = get_encoder(self.configs["text_encoder"])
        self.decoder = get_decoder(self.configs["decoder"])
        self.clap = get_encoder(self.configs["clap"])
        self._freeze_auxiliary_modules()

        # generator
        self.configs["generator"]["schedule"]["output_shape"] = self._latent_shape()
        self.generator = Generator(self.configs["generator"])
        self.generator.set_null_condition(*self.text_encoder.encode_null())
        self.ema = get_ema(self.generator.model, self.configs["training"]["ema"])

        self.heavy_log = False
        self.num_heavy_log_batches = self.configs["training"]["additional_logging"].get("num_heavy_log_batches", 16)

        self.flops_per_sample = measure_generator_flops(
            self.configs["generator"],
            self.configs["text_encoder"]["tokenization_params"]["max_length"],
        )

        self._const_clap_emb = {}
        self.fake_audio_fad_emb = []
        self.train_flops = 0
        self._resamplers = {}
        self.save_hyperparameters({
            "config_id": config_id,
            "configs": json.loads(json.dumps(self.configs, default=str)),
            "flops_per_sample": self.flops_per_sample}
        )

    def _freeze_auxiliary_modules(self):
        for m in (self.audio_encoder, self.text_encoder, self.decoder, self.clap):
            m.eval()
            m.requires_grad_(False)

    @torch.no_grad()
    def _latent_shape(self) -> tuple:
        segment_length = self.configs["training"]["dataset"]["preprocessing_params"]["segment_length"]
        seq_len, latent_dim = self.audio_encoder(torch.randn(1, segment_length)).shape[1:]
        model_params = self.configs["generator"]["model"]
        assert latent_dim == model_params["latent_dim"], (
            f"{self.configs['audio_encoder']['type']} emits {latent_dim}-dim latents, not {model_params['latent_dim']}"
        )
        assert model_params["max_seq_len"] >= seq_len, (
            f"generator max_seq_len {model_params['max_seq_len']} < seq_len {seq_len}"
        )
        return (int(seq_len), int(latent_dim))

    def training_step(self, batch, batch_idx):
        audio, text = batch
        latent, (condition, condition_mask) = self.audio_encoder(audio), self.text_encoder(text)
        del batch, audio, text
        loss = self.generator.training_loss(latent, condition, condition_mask)
        self.log("Training loss", loss)
        self.log_train_flops(latent.shape[0]) 
        self.set_heavy_logging()
        return loss
    
    def validation_step(self, batch, batch_idx):
        audio, text = batch
        condition, condition_mask = self.text_encoder(text)
        if self.heavy_log and batch_idx < self.num_heavy_log_batches:
            self.heavy_logging(condition, condition_mask, audio, text, batch_idx)
        del text
        latent = self.audio_encoder(audio)
        del audio
        loss = self.generator.validation_loss(latent, condition, condition_mask)
        self.log("Validation loss", loss)
        del latent, condition, condition_mask
        return loss

    def heavy_logging(self, condition, condition_mask, audio, text, batch_idx):
        fake_latent = self.generator.sample(condition, condition_mask)
        fake_audio = self.decoder(fake_latent)
        del fake_latent
        fake_audio = self._to_eval_rate(fake_audio, self.decoder.sample_rate)
 
        # AE round-trip ceiling and the real references. Only calculate in first heavy log then cache
        if batch_idx not in self._const_clap_emb:
            recon_audio = self.decoder(self.audio_encoder(audio))
            recon_audio = self._to_eval_rate(recon_audio, self.decoder.sample_rate)
            real_audio = self._to_eval_rate(audio, self.dataset_sample_rate)
            # fake_audio and recon_audio are both self.decoder outputs, so they share a length
            n = min(real_audio.shape[-1], fake_audio.shape[-1])
            self._const_clap_emb[batch_idx] = {
                "real": self.clap.embed_audio(real_audio[..., :n], sr=self.eval_sample_rate).detach(),
                "recon": self.clap.embed_audio(recon_audio[..., :n], sr=self.eval_sample_rate).detach(),
                "text": self.clap.embed_text(text).detach(),
                "length": n,
            }
            if batch_idx == 0:  # constant too, so written once rather than every heavy log
                self.logger.experiment.add_text("Caption", text[0], self.global_step)
                self.register_audio(real_audio[0], label="Real")
                self.register_audio(recon_audio[0], label="Reconstructed")
            del recon_audio, real_audio
        del audio
 
        cached = self._const_clap_emb[batch_idx]
        if batch_idx == 0:
            self.register_audio(fake_audio[0], label="Generated")
 
        fake_clap_emb = self.clap.embed_audio(fake_audio[..., :cached["length"]], sr=self.eval_sample_rate)
        del fake_audio
        self.log("CLAP Cosine Similarity Audio-Text", clap_score(fake_clap_emb, cached["text"]))
        self.log("CLAP Cosine Similarity Recon-Text", clap_score(cached["recon"], cached["text"])) # flat by construction
        self.log("CLAP Cosine Similarity Real-Text", clap_score(cached["real"], cached["text"]))   # flat by construction
        self.fake_audio_fad_emb.append(fake_clap_emb.detach())
        del fake_clap_emb


    def _to_eval_rate(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.eval_sample_rate:
            return audio.float()
        if (sr, self.eval_sample_rate) not in self._resamplers:
            self._resamplers[(sr, self.eval_sample_rate)] = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.eval_sample_rate,
                rolloff=0.99,
                resampling_method="sinc_interp_kaiser",
                lowpass_filter_width=64,
            ).to(audio.device)
        return self._resamplers[(sr, self.eval_sample_rate)](audio).float()

    def register_audio(self, audio: torch.Tensor, label: str = None):
        audio = audio.detach().float().cpu().numpy().squeeze()
        title = f"{label} audio" if label is not None else "Generated audio"
        self.logger.experiment.add_audio(title, audio, self.global_step, sample_rate=self.eval_sample_rate)

    def set_heavy_logging(self):
        interval = self.configs["training"]["additional_logging"]["heavy_logging_interval"]
        if interval is not None and self.global_step % interval == 0:
            self.heavy_log = True

    def log_train_flops(self, batch_size: int):
        self.train_flops += self.flops_per_sample * batch_size * self.trainer.world_size
        self.log("Training PFLOPs", self.train_flops / 1e15)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update()

    def on_before_optimizer_step(self, optimizer):
        if self.configs["training"]["additional_logging"]["log_total_grad_norm"]:
            self.log_total_grad_norm()
        if self.configs["training"]["additional_logging"]["log_detailed_grad_norm"]:
            self.log_detailed_grad_norm()

    def log_total_grad_norm(self):
        self.log("Total grad norm", grad_total_norm(self.generator.model))

    def log_detailed_grad_norm(self):
        norms = grad_norm(self.generator.model, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.generator.model.parameters(), self.configs["training"]["optimizer"])
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
        if self.configs["generator"].get("compile", False):
            self.generator.model = torch.compile(self.generator.model)
        if self.configs["audio_encoder"].get("compile", False):
            self.audio_encoder = torch.compile(self.audio_encoder)
        if self.configs["text_encoder"].get("compile", False):
            self.text_encoder = torch.compile(self.text_encoder)
        if self.configs["decoder"].get("compile", False):
            self.decoder.model = torch.compile(self.decoder.model)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in (self.audio_encoder, self.text_encoder, self.decoder, self.clap):
            m.eval()
        return self

    def on_save_checkpoint(self, checkpoint: dict):
        sd = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith("generator.")}
        checkpoint["state_dict"] = self._remove_compile_ckpt_prefix(sd)
        checkpoint["ema_weights"] = self.ema.state_dict()
        checkpoint["train_flops"] = self.train_flops

    def on_load_checkpoint(self, checkpoint: dict):
        ema_state_dict = checkpoint.get("ema_weights", None)
        if ema_state_dict is not None:
            self.ema.load_state_dict(ema_state_dict)
        self.train_flops = checkpoint.get("train_flops", 0)

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        state_dict = self._remove_compile_ckpt_prefix(state_dict)
        result = super().load_state_dict(state_dict, strict=False)
        missing_generator = [k for k in result.missing_keys if k.startswith("generator.")]
        if strict and (missing_generator or result.unexpected_keys):
            raise RuntimeError(
                "Error loading generator weights.\n"
                f"Missing generator keys: {missing_generator}\n"
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
            if not self.overfit_mode:
                order = sorted(self._const_clap_emb)
                real = self._gather(torch.cat([self._const_clap_emb[i]["real"] for i in order]))
                fake = self._gather(torch.cat(self.fake_audio_fad_emb))
                self.log("FAD CLAP", frechet_distance(real, fake), sync_dist=False)
                recon = self._gather(torch.cat([self._const_clap_emb[i]["recon"] for i in order]))
                self.log("FAD CLAP Reconstructed", frechet_distance(real, recon), sync_dist=False)
            self.fake_audio_fad_emb = []
            self.heavy_log = False

    def _gather(self, x):
        if self.trainer.world_size == 1:
            return x
        return self.all_gather(x).flatten(0, 1)   # (W, N, D) -> (W*N, D)
 
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