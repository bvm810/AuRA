import torch
import math
import copy
from abc import abstractmethod, ABC

class BaseSchedule(ABC):
    def __init__(self, config: dict):
        super().__init__()
        self.sampling_steps = config["sampling_steps"]
        self.output_shape = config["output_shape"]
        self.key_timesteps = config.get("key_timesteps", [0.0, 0.5, 1.0])

    @abstractmethod
    def sample_timesteps(self, num_samples: int) -> torch.Tensor:
        pass

    def training_loss(self, model, target, conditioning) -> torch.Tensor:
        t = self.sample_timesteps(target.shape[0]).to(target.device).to(target.dtype)
        return self.loss(model, target, conditioning, t)

    def validation_loss(self, model, target, conditioning) -> torch.Tensor:
        losses = []
        for t_val in self.key_timesteps:
            t = torch.full((target.shape[0],), t_val, device=target.device, dtype=target.dtype)
            losses.append(self.loss(model, target, conditioning, t))
        return torch.stack(losses).mean()

    @abstractmethod
    def loss(self, model, target, conditioning, timestep) -> torch.Tensor:
        pass

    @abstractmethod
    def sample(self, model, conditioning) -> torch.Tensor:
        pass

    def mse(self, pred, target) -> torch.Tensor:
        diff = (pred - target) ** 2.0               # (B, C, H, W)
        diff = torch.flatten(diff, start_dim=1)     # (B, C * H * W)
        diff = torch.mean(diff, -1)                 # (B,)
        return diff                                 # (B,)

    def channel_broadcast(self, x: torch.Tensor) -> torch.Tensor:
        shape = (-1,) + (1,) * len(self.output_shape)
        return x.reshape(shape)


class RectifiedFlowSchedule(BaseSchedule):
    def __init__(self, config: dict):
        super().__init__(config)
        self.logitnormal_mean = config["logitnormal_mean"]
        self.logitnormal_std = config["logitnormal_std"]

    def sample_timesteps(self, num_samples: int) -> torch.Tensor:
        u = self.logitnormal_mean + self.logitnormal_std * torch.randn(num_samples)
        return torch.sigmoid(u)

    def loss(self, model, target, conditioning, timestep) -> torch.Tensor:
        noise = torch.randn_like(target)
        tb = self.channel_broadcast(timestep)
        noisy = (1.0 - tb) * target + tb * noise
        velocity = noise - target
        pred = model(noisy, timestep, conditioning)
        return self.mse(pred, velocity).mean()

    def sampling_timesteps(self, device, dtype) -> torch.Tensor:
        return torch.linspace(1.0, 0.0, self.sampling_steps + 1, device=device, dtype=dtype)

    def sample(self, model, conditioning) -> torch.Tensor:
        device, dtype = conditioning.device, conditioning.dtype
        ts = self.sampling_timesteps(device, dtype)
        x = torch.randn(conditioning.shape[0], *self.output_shape, device=device, dtype=dtype)
        for i in range(self.sampling_steps):
            t = ts[i] * torch.ones(conditioning.shape[0], device=device, dtype=dtype)
            v = model(x, t, conditioning)
            x = x + (ts[i + 1] - ts[i]) * v
        return x


class ShiftedRectifiedFlowSchedule(RectifiedFlowSchedule):
    def __init__(self, config: dict):
        super().__init__(config)
        self.base_dim = config["base_dim"]
        self.shifted_dim = math.prod(self.output_shape)
        self.alpha = math.sqrt(self.shifted_dim / self.base_dim)
        
    def shift_timestep(self, t: torch.Tensor) -> torch.Tensor:
        t = (self.alpha * t) / (1 + (self.alpha - 1) * t)
        return t

    def loss(self, model, target, conditioning, timestep) -> torch.Tensor:
        timestep = self.shift_timestep(timestep)
        return super().loss(model, target, conditioning, timestep)

    def sampling_timesteps(self, device, dtype) -> torch.Tensor:
        return self.shift_timestep(super().sampling_timesteps(device, dtype))


class ClampedRectifiedFlowSchedule(ShiftedRectifiedFlowSchedule):
    def __init__(self, config: dict):
        super().__init__(config)
        self.mu_min = config["mu_min"]
        self.mu_max = config["mu_max"]
        mu = math.log(self.alpha)
        mu = min(max(mu, self.mu_min), self.mu_max)
        self.alpha = math.exp(mu)

SCHEDULES = {
    "rectified-flow": RectifiedFlowSchedule,
    "shifted-rectified-flow": ShiftedRectifiedFlowSchedule,
    "clamped-rectified-flow": ClampedRectifiedFlowSchedule,
}

def get_schedule(schedule_config: dict):
    config = copy.deepcopy(schedule_config)
    return SCHEDULES[config.pop("type", "rectified-flow")](config)
    