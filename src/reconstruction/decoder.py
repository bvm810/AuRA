import os
import copy
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from diffusers import AutoencoderKL
from transformers import SpeechT5HifiGan
from timm.models.vision_transformer import PatchEmbed
from stable_audio_tools.models import create_model_from_config_path

from src.common.dit import LightningDiTBlock, LightningFinalLayer, VisionRotaryEmbeddingFast, GaussianFourierEmbedding, get_2d_sincos_pos_embed
from src.common.schedule import get_schedule


######################################## Wrappers #############################################

class BaseDecoderWrapper(torch.nn.Module, ABC):
    def __init__(self, decoder_params: dict):
        super().__init__()
        self.model = self._load(**decoder_params["model"])
        self.sample_rate = None

    @abstractmethod
    def _load(self, **kwargs) -> torch.nn.Module:
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z, grad=True)

    def decode(self, z: torch.Tensor, grad: bool = False) -> torch.Tensor:
        if grad:
            decoded = self._decode(z)
        else:
            with torch.no_grad():
                decoded = self._decode(z)
        return decoded

    @abstractmethod
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DiTDecoderWrapper(BaseDecoderWrapper):
    def __init__(self, decoder_params: dict):
        super().__init__(decoder_params)
        self.stft_processor = STFTProcessor(decoder_params["processing"])
        self.schedule = get_schedule(decoder_params["schedule"])
        self.sample_rate = decoder_params["processing"]["sample_rate"]

    def _load(self, checkpoint_path: str, **kwargs) -> torch.nn.Module:
        model = DiTDecoder(**kwargs)
        if checkpoint_path is not None:
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            model.eval()
        return model

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        stft = self.schedule.sample(self.model, z)
        x = self.stft_processor.to_waveform(stft)
        return x


class SAODecoderWrapper(BaseDecoderWrapper):
    def __init__(self, decoder_params: dict):
        super().__init__(decoder_params)
        self.sample_rate = 44100

    def _load(self, checkpoint_path: str) -> torch.nn.Module:
        cfg_path = os.path.join(checkpoint_path, "sao-vae.json")
        ckpt_path = os.path.join(checkpoint_path, "sao-vae.ckpt")
        model = create_model_from_config_path(cfg_path)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
        model.encoder = None
        model.eval()
        return model

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z.transpose(1, 2) # (B, seq_len, D) --> (B, D, seq_len) --> SAO is channels-first
        x = self.model.decode(z) # (B, 2, T)
        x = x.mean(dim=1, keepdim=True) # (B, 1, T)
        return x


class SAMEDecoderWrapper(BaseDecoderWrapper):
    def __init__(self, decoder_params: dict):
        super().__init__(decoder_params)
        self.sample_rate = 44100

    def _load(self, checkpoint_path: str) -> torch.nn.Module:
        cfg_path = os.path.join(checkpoint_path, "same.json")
        ckpt_path = os.path.join(checkpoint_path, "same.ckpt")
        model = create_model_from_config_path(cfg_path)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.encoder = None
        model.eval()
        return model

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z.transpose(1, 2) # (B, seq_len, D) --> (B, D, seq_len) --> SAME is channels-first
        x = self.model.decode(z) # (B, 2, T)
        x = x.mean(dim=1, keepdim=True) # (B, 1, T)
        return x


class AudioLDM2DecoderWrapper(BaseDecoderWrapper):
    def __init__(self, decoder_params: dict):
        super().__init__(decoder_params)
        self.vae = self.model["vae"]
        self.vocoder = self.model["vocoder"]
        self.scaling_factor = self.vae.config.scaling_factor
        self.latent_channels = self.vae.config.latent_channels
        self.sample_rate = 16000
        assert self.sample_rate == self.vocoder.config.sampling_rate, f"Vocoder expects {self.sample_rate} Hz sample rate"

    def _load(self, checkpoint_path: str) -> torch.nn.Module:
        vae = AutoencoderKL.from_pretrained(checkpoint_path, subfolder="vae")
        vae.encoder = None
        vocoder = SpeechT5HifiGan.from_pretrained(checkpoint_path, subfolder="vocoder")
        model = torch.nn.ModuleDict({"vae": vae, "vocoder": vocoder})
        model.eval()
        return model

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, t_lat, d = z.shape
        assert d % self.latent_channels == 0, f"Token dimension {d} does not divide evenly into {self.latent_channels} latent channels"
        f_lat = d // self.latent_channels
        z = z.reshape(batch_size, t_lat, self.latent_channels, f_lat).permute(0, 2, 1, 3).contiguous()
        z = z / self.scaling_factor
        mel = self.vae.decode(z).sample                                # (B, 1, frames, n_mels)
        if mel.dim() == 4:
            mel = mel.squeeze(1)                                       # (B, frames, n_mels)
        audio = self.vocoder(mel)
        return audio.unsqueeze(1).float()                              # (B, 1, T)


################################### pre / post processing #####################################


class STFTProcessor(torch.nn.Module):
    WINDOWS = {
        "hann": torch.hann_window,
        "hamming": torch.hamming_window,
        "blackman": torch.blackman_window,
    }

    def __init__(self, processing_params: dict):
        super().__init__()
        self.processing_params = processing_params
        self.stft_params = dict(processing_params["stft"])
        window_name = self.stft_params.pop("window", None)
        self.register_buffer(
            "window", 
            self.WINDOWS[window_name](self.stft_params["win_length"], periodic=False),
            persistent=False,
        )

    def forward(self, audio: torch.Tensor):
        stft = torch.stft(audio, window=self.window, return_complex=True, **self.stft_params)
        if self.processing_params["compress"]:
            stft = self.compress(stft)
        stft = torch.stack([stft.real, stft.imag], dim=1)
        stft = stft[..., :-1, :] # remove the last frequency bin to make downsampling easier
        return stft

    def compress(self, stft: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(stft)
        phase = torch.angle(stft)
        beta = self.processing_params["compression"]["magnitude_scaling_factor"]
        alpha = self.processing_params["compression"]["magnitude_compression_factor"]
        return beta * torch.pow(mag, alpha) * torch.exp(1j * phase)

    # stft here has shape (batch, channels, nfft // 2, frames)
    def to_waveform(self, stft: torch.Tensor) -> torch.Tensor:
        real, img = torch.chunk(stft.float(), chunks=2, dim=-3)
        stft = torch.complex(real.squeeze(1), img.squeeze(1))
        if self.processing_params["compress"]:
            stft = self.decompress(stft)
        zero_bin = torch.zeros_like(stft[..., :1, :])           # (B, 1, T) complex
        stft = torch.cat([stft, zero_bin], dim=-2)              # (B, nfft//2 + 1, T) complex
        waveform = torch.istft(
            stft, 
            window=self.window,
            return_complex=False, 
            length=self.processing_params["segment_length"],
            **self.stft_params
        )
        return waveform.unsqueeze(1) # (batch, 1, samples) --> for torchaudio save we need channels

    def decompress(self, stft: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(stft)
        phase = torch.angle(stft)
        beta = self.processing_params["compression"]["magnitude_scaling_factor"]
        alpha = self.processing_params["compression"]["magnitude_compression_factor"]
        return torch.pow(mag * (1 / beta), 1 / alpha) * torch.exp(1j * phase)


######################################## DiT Decoder ##########################################

class DiTDecoder(nn.Module):
    def __init__(
        self,
        input_size: tuple,
        patch_size: tuple,
        hidden_size: int,
        frontend_bias: bool,
        context_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: int,
        use_qknorm: bool,
        use_swiglu: bool,
        use_rope: bool,
        use_rmsnorm: bool,
        wo_shift: bool,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.frontend_bias = frontend_bias
        self.context_dim = context_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_qknorm = use_qknorm
        self.use_swiglu = use_swiglu
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.wo_shift = wo_shift

        self.frontend = ScaledPatchFrontend(self.input_size, self.patch_size, self.hidden_size, self.frontend_bias)
        self.t_embedder = GaussianFourierEmbedding(self.hidden_size)
        
        num_patches = self.frontend.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.hidden_size), requires_grad=False)
        if self.use_rope:
            half_head_dim = self.hidden_size // self.num_heads // 2
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=self.frontend.grid_size,
            )
        else:
            self.feat_rope = None

        self.blocks = nn.ModuleList([
            LightningDiTBlock(
                self.hidden_size, 
                self.context_dim,
                self.num_heads, 
                mlp_ratio=self.mlp_ratio, 
                use_qknorm=self.use_qknorm, 
                use_swiglu=self.use_swiglu, 
                use_rmsnorm=self.use_rmsnorm,
                wo_shift=self.wo_shift,

            ) for _ in range(self.depth)
        ])

        self.final_layer = LightningFinalLayer(self.hidden_size, self.frontend.patch_size, self.frontend.out_channels, use_rmsnorm=self.use_rmsnorm)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.frontend.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)    

    def forward(self, x, t, context):
        """
        Forward pass of LightningDiT.
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B,) tensor of diffusion timesteps
        context: (B, Nt, Nf * D) tensor of SSL features
        use_checkpoint: boolean to toggle checkpointing
        """
        t = self.t_embedder(t)                                              # (N, D)
        context = context.reshape(context.shape[0], -1, self.context_dim)   # (B, Nt, Nf * D) --> (B, Nt * Nf, D))

        x = self.frontend.patchify(x, t)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x, t, context, feat_rope=self.feat_rope)

        x = self.final_layer(x, t)                                          # (N, T, patch_size ** 2 * out_channels)
        x = self.frontend.unpatchify(x)                                     # (N, out_channels, H, W)

        return x


class ScaledPatchFrontend(torch.nn.Module):
    def __init__(self, input_size, patch_size, hidden_size, bias):
        super().__init__()
        self.input_size = input_size
        self.patchifier = PatchEmbed(input_size, patch_size, 2, hidden_size, bias=bias)
        w = self.patchifier.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.patchifier.proj.bias is not None:
            torch.nn.init.constant_(self.patchifier.proj.bias, 0)
        self.timestep_scaler = torch.nn.Linear(hidden_size, input_size[0])
        self.grid_size = self.patchifier.grid_size
        self.patch_size = self.patchifier.patch_size
        self.num_patches = self.patchifier.num_patches
        self.out_channels = 2 # stft always has 2 channels

    # just for torch compatibility, in practice I'll call patchify or unpatchify
    def forward(self, x, t_emb):
        return self.patchify(x, t_emb)

    def patchify(self, x, t_emb):
        scale = self.timestep_scaler(t_emb)
        x = x * scale.view(-1, 1, self.input_size[0], 1)
        return self.patchifier(x)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size[0] * patch_size[1] * 2)
        imgs: (N, 2, H, W)
        """
        c = self.out_channels 
        ph, pw = self.patchifier.patch_size
        gh, gw = self.patchifier.grid_size
        assert gh * gw == x.shape[1]

        x = x.reshape(shape=(x.shape[0], gh, gw, ph, pw, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, gh * ph, gw * pw))                                         
        return imgs


################################################# Registry ########################################


DECODER_REGISTRY = {
    "dit": DiTDecoderWrapper,
    "sao": SAODecoderWrapper,
    "same": SAMEDecoderWrapper,
    "audioldm2": AudioLDM2DecoderWrapper,
}

def get_decoder(config: dict) -> BaseDecoderWrapper:
    cfg = copy.deepcopy(config)
    decoder_type = cfg.pop("type")
    return DECODER_REGISTRY[decoder_type](cfg)