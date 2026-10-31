import torch
import torch.nn as nn

from functools import partial
from typing import Callable
from src.common.schedule import get_schedule
from src.common.dit import LightningDiTBlock, LightningFinalLayer, GaussianFourierEmbedding, SinCosPosEmbedding1D, RotaryEmbedding1D

############################### Wrapper ###############################

class Generator(torch.nn.Module):
    def __init__(self, generator_params: dict):
        super().__init__()
        self.model = GeneratorDiT(**generator_params["model"])
        self.schedule = get_schedule(generator_params["schedule"])
        self.latent_mean = generator_params["latent_mean"]
        self.latent_std = generator_params["latent_std"]
        self.cfg_dropout_prob = generator_params["cfg_dropout_prob"]
        self.cfg_scale = generator_params["cfg_scale"]
        self.register_buffer("null_condition", None, persistent=False)
        self.register_buffer("null_condition_mask", None, persistent=False)

    def set_null_condition(self, condition: torch.Tensor, condition_mask: torch.Tensor):
        self.null_condition = condition.detach()
        self.null_condition_mask = condition_mask.detach()

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.latent_mean) / self.latent_std

    def unnormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.latent_std + self.latent_mean

    def forward(self, z: torch.Tensor, timestep: torch.Tensor, conditioning: torch.Tensor, conditioning_mask: torch.Tensor = None) -> torch.Tensor:
        return self.model(z, timestep, conditioning, conditioning_mask)

    def _null_like(self, conditioning: torch.Tensor, conditioning_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.null_condition is not None, "CFG requires a null condition"
        assert self.null_condition.shape[1] == conditioning.shape[1], "Null condition and conditioning must have the same number of tokens"
        null = self.null_condition.to(conditioning.dtype).expand_as(conditioning)
        null_mask = self.null_condition_mask.to(conditioning_mask.dtype).expand_as(conditioning_mask)
        return null, null_mask
 
    def cfg_dropout(self, conditioning: torch.Tensor, conditioning_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        null, null_mask = self._null_like(conditioning, conditioning_mask)
        drop = torch.rand(conditioning.shape[0], device=conditioning.device) < self.cfg_dropout_prob  # (B,)
        conditioning = torch.where(drop[:, None, None], null, conditioning)
        conditioning_mask = torch.where(drop[:, None], null_mask, conditioning_mask)
        return conditioning, conditioning_mask

    def _bind(self, conditioning_mask: torch.Tensor) -> Callable:
        return self.model if conditioning_mask is None else partial(self.model, context_mask=conditioning_mask)

    # same as _bind but placing null condition in the batch dimension to avoid two sampling passes
    def _bind_guided(self, conditioning: torch.Tensor, conditioning_mask: torch.Tensor) -> Callable:
        null, null_mask = self._null_like(conditioning, conditioning_mask)
        context = torch.cat([conditioning, null], dim=0)
        context_mask = torch.cat([conditioning_mask, null_mask], dim=0)
 
        def guided(x, t, _conditioning):
            v = self.model(torch.cat([x, x], dim=0), torch.cat([t, t], dim=0), context, context_mask=context_mask)
            v_cond, v_uncond = v.chunk(2, dim=0)
            return v_uncond + self.cfg_scale * (v_cond - v_uncond)
 
        return guided

    def training_loss(self, target: torch.Tensor, conditioning: torch.Tensor, conditioning_mask: torch.Tensor) -> torch.Tensor:
        target = self.normalize(target)
        conditioning, conditioning_mask = self.cfg_dropout(conditioning, conditioning_mask)
        loss = self.schedule.training_loss(self._bind(conditioning_mask), target, conditioning)
        return loss

    def validation_loss(self, target: torch.Tensor, conditioning: torch.Tensor, conditioning_mask: torch.Tensor) -> torch.Tensor:
        target = self.normalize(target)
        loss = self.schedule.validation_loss(self._bind(conditioning_mask), target, conditioning)
        return loss

    @torch.no_grad()
    def sample(self, conditioning: torch.Tensor, conditioning_mask: torch.Tensor) -> torch.Tensor:
        call = self._bind(conditioning_mask) if self.cfg_scale == 1.0 else self._bind_guided(conditioning, conditioning_mask)
        z = self.schedule.sample(call, conditioning)
        return self.unnormalize(z)


################################ Generator ###########################

class GeneratorDiT(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        latent_dim: int,
        context_dim: int,
        max_seq_len: int,
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
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_qknorm = use_qknorm
        self.use_swiglu = use_swiglu
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.wo_shift = wo_shift

        self.frontend = torch.nn.Linear(self.latent_dim, self.hidden_size)
        self.t_embedder = GaussianFourierEmbedding(self.hidden_size)
        self.pos_embed = SinCosPosEmbedding1D(embed_dim=self.hidden_size, max_len=self.max_seq_len)

        if self.use_rope:
            head_dim = self.hidden_size // self.num_heads
            self.feat_rope = RotaryEmbedding1D(dim=head_dim, max_seq_len=self.max_seq_len)
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
        self.final_layer = LightningFinalLayer(self.hidden_size, (1, 1), self.latent_dim, use_rmsnorm=self.use_rmsnorm)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

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

    def forward(self, x, t=None, context=None, context_mask=None):
        t = self.t_embedder(t) # (B, D)

        x = self.frontend(x)  # (B, N, D) --> project the input to the hidden size
        x = self.pos_embed(x) # (B, N, D) --> forward on this block does the addition

        for block in self.blocks:
            x = block(x, t, context, feat_rope=self.feat_rope, context_mask=context_mask)

        x = self.final_layer(x, t) # (B, N, latent_dim) --> project the output to the latent dimension
        return x