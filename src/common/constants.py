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
        "sample_rate": SSL_SAMPLE_RATE
    },
    "beats": {
        "type": "beats",
        "model": {"checkpoint_path": f"{BEATS_DIR}/beats_iter3_pretrained_encoder.pt"},
        "sample_rate": SSL_SAMPLE_RATE
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
}


######################################## DATA #####################################################


AUDIOSET_CONFIG = {
    "root_dir": AUDIOSET_DIR,
    "csv_name": "metadata.csv",
    "audio_dir_name": "audio",
    "audio_column": "filepath",
    "split_column": "split",
    "preprocessing_params": {
        "sample_rate": DATASET_SAMPLE_RATES["audioset"],
        "segment_length": SEGMENT_LENGTH_16K
    }
}

WAVCAPS_CONFIG = {
    "root_dir": WAVCAPS_DIR,
    "csv_name": "long-segments.csv",
    "audio_dir_name": "audio",
    "audio_column": "filepath",
    "caption_column": "caption",
    "split_column": "split",
    "preprocessing_params": {
        "sample_rate": DATASET_SAMPLE_RATES["wavcaps"],
        "segment_length": SEGMENT_LENGTH_32K
    }
}

AUDIOCAPS_EVAL_CONFIG = {
    "root_dir": AUDIOCAPS_DIR,
    "csv_name": "metadata.csv",
    "audio_dir_name": "audio",
    "audio_column": "filepath",
    "split_column": "split",
    "preprocessing_params": {
        "sample_rate": DATASET_SAMPLE_RATES["audiocaps"],
        "segment_length": SEGMENT_LENGTH_16K
    },
    "return_path": True,
}

WAVCAPS_EVAL_CONFIG = {
    "root_dir": WAVCAPS_DIR,
    "csv_name": "long-segments.csv",
    "audio_dir_name": "audio",
    "audio_column": "filepath",
    "caption_column": "caption",
    "split_column": "split",
    "preprocessing_params": {
        "sample_rate": DATASET_SAMPLE_RATES["wavcaps"],
        "segment_length": SEGMENT_LENGTH_32K
    },
    "return_path": True,
}

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


def encoder_config(encoder: str):
    base_conf = copy.deepcopy(ENCODER_DEFAULT_CONFIGS[encoder])
    base_conf["resampling"] = {
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
