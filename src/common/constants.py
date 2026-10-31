import os
import copy

from pathlib import Path
from typing import Optional


############################################ PATHS ################################################


HOME = Path(os.environ.get("HOME", ""))

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = ROOT / "checkpoints"
LOGS_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"

MATPAC_DIR = CHECKPOINT_DIR / "matpac"
BEATS_DIR = CHECKPOINT_DIR / "beats"
SAO_DIR = CHECKPOINT_DIR / "sao"
SAME_DIR = CHECKPOINT_DIR / "same"
AUDIOLDM2_DIR = CHECKPOINT_DIR / "audioldm2"

AUDIOSET_DIR = HOME / "data" / "audioset"
AUDIOCAPS_DIR = HOME / "data" / "audiocaps"
WAVCAPS_DIR = HOME / "data" / "captioning"


###################################### CONSTANTS ################################################


DATASET_SAMPLE_RATES = {
    "audioset": 16_000,
    "wavcaps": 32_000,
    "audiocaps": 16_000,
    "wavcaps-eval": 32_000,
}

SSL_SAMPLE_RATE = 16_000
EVAL_SAMPLE_RATE = 16_000
SEGMENT_LENGTH_16K = 159_488
SEGMENT_LENGTH_32K = 318_976


##################################### ENCODER DEFAULTS ##########################################


ENCODER_DEFAULT_CONFIGS = {
    "matpac": {
        "type": "matpac",
        "model": {
            "checkpoint_path": f"{MATPAC_DIR}/matpac_plus_as_48_1_map_enconly.pt",
            "inference_type": "precise",
            "pull_time_dimension": False,
            "concat_freq": True,
        },
        "input_sample_rate": SSL_SAMPLE_RATE
    },
    "beats": {
        "type": "beats",
        "model": {"checkpoint_path": f"{BEATS_DIR}/beats_iter3_pretrained_encoder.pt"},
        "input_sample_rate": SSL_SAMPLE_RATE
    },
    "sao": {
        "type": "sao",
        "model": {"checkpoint_path": f"{SAO_DIR}"},
        "input_sample_rate": 44_100
    },
    "same": {
        "type": "same",
        "model": {"checkpoint_path": f"{SAME_DIR}"},
        "input_sample_rate": 44_100
    },
    "audioldm2": {
        "type": "audioldm2",
        "model": {"checkpoint_path": f"{AUDIOLDM2_DIR}"},
        "mel": {
            "sample_rate": 16_000,
            "n_fft": 1024,
            "hop_length": 160,
            "win_length": 1024,
            "n_mels": 64,
            "fmin": 0,
            "fmax": 8000,
        },
        "input_sample_rate": 16_000
    },
}


#################################### DECODER DEFAULTS #############################################


DIT_DECODER_DEFAULT_CONFIG = {
    "type": "dit",
    "model": {
        "input_size": (512, 624),
        "hidden_size": 1024,
        "patch_size": (16, 8),
        "frontend_bias": True,
        "context_dim": 768,
        "depth": 12,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "use_qknorm": True,
        "use_swiglu": True,
        "use_rope": True,
        "use_rmsnorm": True,
        "wo_shift": False,
    },
    "schedule": {
        "type": "rectified-flow",
        "sampling_steps": 50,
        "logitnormal_mean": 0.0,
        "logitnormal_std": 1.0,
        "output_shape": (2, 512, 624)
    },
    "processing": {
        "stft": {
            "n_fft": 1024,
            "win_length": 1024,
            "hop_length": 256,
            "window": "hann",
            "center": True,
            "normalized": False,
            "onesided": True,
        },
        "compress": True,
        "compression": {
            "magnitude_compression_factor": 0.65,
            "magnitude_scaling_factor": 0.34,
        },
        "sample_rate": SSL_SAMPLE_RATE,
        "segment_length": SEGMENT_LENGTH_16K,
    }
}

DECODER_DEFAULT_CONFIGS = {
    "matpac": DIT_DECODER_DEFAULT_CONFIG,
    "beats": DIT_DECODER_DEFAULT_CONFIG,
    "sao": {"type": "sao", "model": {"checkpoint_path": f"{SAO_DIR}"}},
    "same": {"type": "same", "model": {"checkpoint_path": f"{SAME_DIR}"}},
    "audioldm2": {"type": "audioldm2", "model": {"checkpoint_path": f"{AUDIOLDM2_DIR}"}},
}


############################### T5 AND CLAP DEFAULTS ####################################


TEXT_ENCODER_DEFAULT_CONFIG = {
    "type": "t5",
    "model": {"checkpoint_path": "google-t5/t5-base"},
    "tokenization_params": {
        "padding": "max_length",
        "max_length": 64,
        "truncation": True,
    },
}

CLAP_DEFAULT_CONFIG = {
    "type": "clap",
    "model": {"checkpoint_path": "laion/clap-htsat-unfused"},
    "feature_extraction": {"truncation": "rand_trunc", "padding": "repeatpad"},
    "tokenization_params": {"padding": True, "truncation": True},
    "max_duration": 10.0,
}


######################################## DATA #####################################################


DEFAULT_DATALOADERS = {
    "train": {
        "persistent_workers": True,
        "pin_memory": True,
        "drop_last": True,
        "shuffle": True,    
    },
    "validation": {
        "persistent_workers": True,
        "pin_memory": True,
        "drop_last": False,
        "shuffle": False,
    },
}


######################################## TRAINING #################################################

DEFAULT_CHECKPOINTING = [
    {
        "save_last": False,
        "enable_version_counter": False,      
        "every_n_epochs": None,
        "save_top_k": -1,                                    
    },
    {
        "filename": "last",
        "monitor": None,
        "save_top_k": 1,
        "every_n_epochs": 1,
        "save_on_train_epoch_end": True,
        "save_last": False,
        "enable_version_counter": False,
    }
]


######################################## BUILDERS #################################################


def encoder_config(encoder: str, dataset: str):
    base_conf = copy.deepcopy(ENCODER_DEFAULT_CONFIGS[encoder])
    target_sr = base_conf.pop("input_sample_rate")
    base_conf["resampler"] = {
        "orig_freq": DATASET_SAMPLE_RATES[dataset],
        "new_freq": target_sr,
        "rolloff": 0.99,
        "resampling_method": "sinc_interp_kaiser",
        "lowpass_filter_width": 64,
    }
    return base_conf


def decoder_config(encoder: str, ckpt_path: Optional[str] = None):
    base_conf = copy.deepcopy(DECODER_DEFAULT_CONFIGS[encoder])
    base_conf["model"]["checkpoint_path"] = ckpt_path
    return base_conf

def loader_config(batch_size: int = 1, num_workers: int = 6):
    base_conf = copy.deepcopy(DEFAULT_DATALOADERS)
    base_conf["train"]["batch_size"] = batch_size
    base_conf["train"]["num_workers"] = num_workers
    base_conf["validation"]["batch_size"] = batch_size
    base_conf["validation"]["num_workers"] = num_workers
    return base_conf

def checkpointing_config(interval: int = 1000, save_dir: Path = CHECKPOINT_DIR):
    base_conf = copy.deepcopy(DEFAULT_CHECKPOINTING)
    for c in base_conf:
        c["dirpath"] = save_dir
    base_conf[0]["every_n_train_steps"] = interval
    return base_conf