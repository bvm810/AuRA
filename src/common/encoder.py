import os
import torch
import torchaudio

from abc import abstractmethod, ABC
from diffusers import AutoencoderKL
from matpac.model import get_matpac
from librosa.filters import mel as librosa_mel_fn
from stable_audio_tools.models import create_model_from_config_path
from transformers import AutoTokenizer, ClapModel, ClapProcessor, T5EncoderModel

from src.common.beats.BEATs import BEATs, BEATsConfig


##################################### Audio Encoders #####################################


class BaseAudioEncoderWrapper(torch.nn.Module, ABC):
    def __init__(self, encoder_params: dict):
        super().__init__()
        self.resampler = torchaudio.transforms.Resample(**encoder_params["resampler"])
        self.model = self._load(**encoder_params["model"])

    @abstractmethod
    def _load(self, **kwargs) -> torch.nn.Module:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resampler(x)
        return self._encode(x)

    @abstractmethod
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MATPACEncoderWrapper(BaseAudioEncoderWrapper):
    def __init__(self, encoder_params: dict):
        super().__init__(encoder_params)
        self.embed_dim = int(self.model.cfg.encoder.embed_dim)
        self.n_freq_patches = int(self.model.grid_size()[0])

    def _load(self, **kwargs) -> torch.nn.Module:
        model = get_matpac(**kwargs)
        model.eval()
        return model

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        emb, _ = self.model(x)                                  # (B, Nt, Nf * D)
        assert emb.shape[-1] == self.n_freq_patches * self.embed_dim, (
            f"expected a last dim of {self.n_freq_patches * self.embed_dim}, got {emb.shape[-1]}"
        )
        return emb.reshape(emb.shape[0], -1, self.embed_dim)    # (B, Nt * Nf, D)


class BEATsEncoderWrapper(BaseAudioEncoderWrapper):
    def __init__(self, encoder_params: dict):
        super().__init__(encoder_params)

    def _load(self, checkpoint_path: str) -> torch.nn.Module:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = BEATsConfig(checkpoint["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        return model

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(x).bool()
        emb, _ = self.model.extract_features(x, padding_mask=mask)
        return emb


class SAOEncoderWrapper(BaseAudioEncoderWrapper):
    def __init__(self, encoder_params: dict):
        super().__init__(encoder_params)

    def _load(self, checkpoint_path: str) -> torch.nn.Module:
        cfg_path = os.path.join(checkpoint_path, "sao-vae.json")
        ckpt_path = os.path.join(checkpoint_path, "sao-vae.ckpt")
        model = create_model_from_config_path(cfg_path)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
        model.decoder = None
        model.eval()
        return model

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1).repeat(1, 2, 1)  # (B, T) -> (B, 1, T) -> (B, 2, T)
        emb = self.model.encode(x)          # (B, D, seq_len) --> SAO is channels-first
        return emb.transpose(1, 2)          # (B, D, seq_len) --> (B, seq_len, D)


class SAMEEncoderWrapper(BaseAudioEncoderWrapper):
    def __init__(self, encoder_params: dict):
        super().__init__(encoder_params)

    def _load(self, checkpoint_path: str) -> torch.nn.Module:
        cfg_path = os.path.join(checkpoint_path, "same.json")
        ckpt_path = os.path.join(checkpoint_path, "same.ckpt")
        model = create_model_from_config_path(cfg_path)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.decoder = None
        model.eval()
        return model

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        ratio = self.model.downsampling_ratio
        remainder = x.shape[-1] % ratio
        if remainder:
            x = torch.nn.functional.pad(x, (0, ratio - remainder))
        x = x.unsqueeze(1).repeat(1, 2, 1)  # (B, T) -> (B, 1, T) -> (B, 2, T)
        emb = self.model.encode(x)          # (B, D, seq_len) --> SAME is channels-first
        return emb.transpose(1, 2)          # (B, D, seq_len) --> (B, seq_len, D)


# AudioLDM2 mel-spectrogram frontend
class AudioLDM2Mel(torch.nn.Module):
    def __init__(self, mel_params: dict, downsample_factor: int = 4):
        super().__init__()
        self.mel_params = mel_params
        self.downsample_factor = downsample_factor
        mel_basis = librosa_mel_fn(
            sr=mel_params["sample_rate"],
            n_fft=mel_params["n_fft"],
            n_mels=mel_params["n_mels"],
            fmin=mel_params["fmin"],
            fmax=mel_params["fmax"],
        )
        self.register_buffer("window", torch.hann_window(mel_params["win_length"]), persistent=False)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float(), persistent=False)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=-1, keepdim=True)
        peak = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        return x / peak * 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize(x.float()).clamp(-1.0, 1.0)       # (B, T)
        stft = torch.stft(
            x,
            n_fft=self.mel_params["n_fft"],
            hop_length=self.mel_params["hop_length"],
            win_length=self.mel_params["win_length"],
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        mag = stft.abs()
        mel = torch.matmul(self.mel_basis, mag)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        mel = mel.transpose(1, 2)
        pad = (-mel.shape[1]) % self.downsample_factor
        if pad > 0:
            mel = torch.nn.functional.pad(mel, (0, 0, 0, pad))
        return mel


class AudioLDM2EncoderWrapper(BaseAudioEncoderWrapper):
    def __init__(self, encoder_params: dict):
        super().__init__(encoder_params)
        self.scaling_factor = self.model.config.scaling_factor
        self.latent_channels = self.model.config.latent_channels
        assert encoder_params["mel"]["sample_rate"] == encoder_params["resampler"]["new_freq"], (
            f"AudioLDM2 mel requires {encoder_params['mel']['sample_rate']} Hz, got {encoder_params['resampler']['new_freq']} Hz"
        )
        downsample_factor = 2 ** (len(self.model.config.block_out_channels) - 1)
        self.mel = AudioLDM2Mel(encoder_params["mel"], downsample_factor)

    def _load(self, checkpoint_path: str) -> torch.nn.Module:
        model = AutoencoderKL.from_pretrained(checkpoint_path, subfolder="vae")
        model.decoder = None
        model.eval()
        return model

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()                                    # (B, T) avoiding autocast
        mel = self.mel(x).unsqueeze(1)                   # (B, 1, frames, n_mels)
        z = self.model.encode(mel).latent_dist.sample()  # (B, C, T_lat, F_lat)
        z = z * self.scaling_factor
        batch_size, channels, t_lat, f_lat = z.shape
        assert channels == self.latent_channels, f"expected {self.latent_channels} latent channels, got {channels}"
        return z.permute(0, 2, 1, 3).reshape(batch_size, t_lat, channels * f_lat)  


##################################### Text Encoders #######################################


class BaseTextEncoderWrapper(torch.nn.Module, ABC):
    def __init__(self, encoder_params: dict):
        super().__init__()
        self.model, self.tokenizer = self._load(**encoder_params["model"])
        self.tokenization_params = encoder_params["tokenization_params"]
        self.hidden_size = None

    @abstractmethod
    def _load(self, **kwargs) -> tuple[torch.nn.Module, AutoTokenizer]:
        raise NotImplementedError

    def forward(self, x: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(x)

    @torch.no_grad()
    def encode(self, x: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, attention_mask = self.tokenize(x)
        return self._encode(input_ids, attention_mask), attention_mask

    def tokenize(self, x: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.tokenizer(x, return_tensors="pt", **self.tokenization_params)
        device = self.model.device  # tokenizer always returns CPU tensors
        return batch["input_ids"].to(device), batch["attention_mask"].to(device)

    def encode_null(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode([""])

    @abstractmethod
    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class T5EncoderWrapper(BaseTextEncoderWrapper):
    def __init__(self, encoder_params: dict):
        super().__init__(encoder_params)
        self.hidden_size = self.model.config.d_model

    def _load(self, checkpoint_path: str) -> torch.nn.Module:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = T5EncoderModel.from_pretrained(checkpoint_path)
        model.eval()
        return model, tokenizer

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state


##################################### CLAP Encoder #########################################


class CLAPWrapper(torch.nn.Module):
    def __init__(self, encoder_params: dict):
        super().__init__()
        self.model, self.processor = self._load(**encoder_params["model"])
        self.sample_rate = self.processor.feature_extractor.sampling_rate
        self.feature_extraction_params = encoder_params["feature_extraction"]
        self.tokenization_params = encoder_params["tokenization_params"]
        self.max_length = int(encoder_params["max_duration"] * self.sample_rate)
        self._resamplers = {}  # plain dict: kept out of state_dict, built on the right device
 
    def _load(self, checkpoint_path: str) -> tuple[ClapModel, ClapProcessor]:
        processor = ClapProcessor.from_pretrained(checkpoint_path)
        model = ClapModel.from_pretrained(checkpoint_path)
        model.eval()
        return model, processor
 
    def _resample(self, x: torch.Tensor, sr: int) -> torch.Tensor:
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(
                orig_freq=sr, 
                new_freq=self.sample_rate,
                resampling_method="sinc_interp_kaiser",
                lowpass_filter_width=64,
            ).to(x.device)
        return self._resamplers[sr](x)
 
    @torch.no_grad()
    def embed_audio(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        if audio.dim() == 3:  # (B, C, T) --> mono
            audio = audio.mean(dim=1)
        audio = self._resample(audio, sr).float()
        # CLAP random crops audio longer than 10s at its native sample rate. 
        # I'm truncating, as my files are shorter and exceeding samples are padding / resampling artifacts.
        if audio.shape[-1] > self.max_length:
            audio = audio[..., :self.max_length]
        # Generated audio is not guaranteed to be in range, so clamp to match.
        audio = audio.clamp(-1.0, 1.0)
        inputs = self.processor(
            audios=list(audio.cpu().numpy()),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            **self.feature_extraction_params,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return self.model.get_audio_features(**inputs)
 
    @torch.no_grad()
    def embed_text(self, text: list[str]) -> torch.Tensor:
        inputs = self.processor(text=text, return_tensors="pt", **self.tokenization_params)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return self.model.get_text_features(**inputs)


##################################### Encoder Registry #####################################


ENCODER_REGISTRY = {
    "matpac": MATPACEncoderWrapper,
    "beats": BEATsEncoderWrapper,
    "sao": SAOEncoderWrapper,
    "same": SAMEEncoderWrapper,
    "audioldm2": AudioLDM2EncoderWrapper,
    "t5": T5EncoderWrapper,
    "clap": CLAPWrapper,
}

def get_encoder(config: dict):
    cfg = config.copy()
    encoder_type = cfg.pop("type")
    return ENCODER_REGISTRY[encoder_type](cfg)