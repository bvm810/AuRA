from src.common.constants import *


WAVCAPS_CONFIG = {  # WavCaps + AudioCaps train / validation splits
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
    "caption_column": "caption",
    "split_column": "split",
    "preprocessing_params": WAVCAPS_CONFIG["preprocessing_params"],
    "return_path": True,
}

WAVCAPS_EVAL_CONFIG = {
    "root_dir": WAVCAPS_DIR,
    "csv_name": "isolated-eval.csv",
    "audio_dir_name": "audio",
    "audio_column": "filepath",
    "caption_column": "caption",
    "split_column": "split",
    "preprocessing_params": WAVCAPS_CONFIG["preprocessing_params"],
    "return_path": True,
}

GENERATOR_MODEL = {
    "hidden_size": 1024,
    "context_dim": 768,  # T5-base hidden size
    "max_seq_len": 1024,
    "depth": 12,
    "num_heads": 16,
    "mlp_ratio": 4.0,
    "use_qknorm": True,
    "use_swiglu": True,
    "use_rope": True,
    "use_rmsnorm": True,
    "wo_shift": False,
}

GENERATOR_SCHEDULE = {  # output_shape is set by the system from the encoder output
    "type": "clamped-rectified-flow",
    "sampling_steps": 50,
    "logitnormal_mean": 0.0,
    "logitnormal_std": 1.0,
    "base_dim": 4096,
    "mu_min": 0.5,
    "mu_max": 1.15,
}

CHECKPOINT_INTERVAL = 9_419  # one pass over the corpus: (6645 h * 3600 s * 32 kHz // 317,440-sample segments) // 256 global batch

BASE_GENERATION_TRAINING = {
    "ema": {
        "decay": 0.9999,
        "use_num_updates": True,
    },
    "optimizer": {
        "type": "RAdam",
        "params": {
            "lr": 1e-4,
        }
    },
    "lr_scheduler": {
        "type": "ConstantLR",
        "params": {
            "factor": 1.0,
            "total_iters": 188_380,
        }
    },
    "strategy": {
        "type": "DDP",
        "params": {
            "gradient_as_bucket_view": True,
            "find_unused_parameters": False
        },
    },
    "trainer": {
        "max_epochs": -1,
        "max_steps": 188_380,
        "num_nodes": 1,
        "devices": 4,
        "precision": "bf16-mixed",
        "check_val_every_n_epoch": None,  # val_check_interval spans epochs
    },
    "additional_logging": {
        "heavy_logging_interval": 9_419,
        "log_total_grad_norm": True,
        "log_detailed_grad_norm": False,
    },
    "dataset": WAVCAPS_CONFIG
}

# Settings per GPU batch size, keeping a global batch of 256 on 4 GPUs and 1024 FAD clips per heavy log.
# val_check_interval counts batches, so it is the heavy logging interval times the accumulation.

BATCH_SETTINGS = {
    64: {
        "dataloaders": loader_config(batch_size=64, num_workers=10),
        "trainer": {**BASE_GENERATION_TRAINING["trainer"], "accumulate_grad_batches": 1, "val_check_interval": 9_419},
        "additional_logging": {**BASE_GENERATION_TRAINING["additional_logging"], "num_heavy_log_batches": 4},
    },
    32: {
        "dataloaders": loader_config(batch_size=32, num_workers=10),
        "trainer": {**BASE_GENERATION_TRAINING["trainer"], "accumulate_grad_batches": 2, "val_check_interval": 18_838},
        "additional_logging": {**BASE_GENERATION_TRAINING["additional_logging"], "num_heavy_log_batches": 8},
    },
    16: {
        "dataloaders": loader_config(batch_size=16, num_workers=10),
        "trainer": {**BASE_GENERATION_TRAINING["trainer"], "accumulate_grad_batches": 4, "val_check_interval": 37_676},
        "additional_logging": {**BASE_GENERATION_TRAINING["additional_logging"], "num_heavy_log_batches": 16},
    },
    8: {
        "dataloaders": loader_config(batch_size=8, num_workers=10),
        "trainer": {**BASE_GENERATION_TRAINING["trainer"], "accumulate_grad_batches": 8, "val_check_interval": 75_352},
        "additional_logging": {**BASE_GENERATION_TRAINING["additional_logging"], "num_heavy_log_batches": 32},
    },
}

CONFIGS = {
    "generation-matpac": {
        "audio_encoder": encoder_config("matpac", "wavcaps"),
        "text_encoder": TEXT_ENCODER_DEFAULT_CONFIG,
        "decoder": decoder_config("matpac", CHECKPOINT_DIR / "reconstruction-matpac" / "decoder.pt"),
        "clap": CLAP_DEFAULT_CONFIG,
        "generator": {
            "model": {**GENERATOR_MODEL, "latent_dim": 768},
            "schedule": GENERATOR_SCHEDULE,
            "latent_mean": -0.0281093,
            "latent_std": 0.164946,
            "cfg_dropout_prob": 0.1,
            "cfg_scale": 5.0,
        },
        "training": {
            **BASE_GENERATION_TRAINING,
            **BATCH_SETTINGS[64],
            "logger": {"save_dir": LOGS_DIR / "generation-matpac"},
            "checkpointing": checkpointing_config(interval=CHECKPOINT_INTERVAL, save_dir=CHECKPOINT_DIR / "generation-matpac"),
            "checkpoint": CHECKPOINT_DIR / "generation-matpac" / "last.ckpt",
        },
        "evaluation": {
            "checkpoint": CHECKPOINT_DIR / "generation-matpac" / "last.ckpt",  # placeholder, point to the checkpoint to evaluate
            "dataset": AUDIOCAPS_EVAL_CONFIG,
            "dataloaders": BATCH_SETTINGS[64]["dataloaders"]["validation"],
            "save": {
                "save_dir": RESULTS_DIR / "generation-matpac" / "audiocaps",
                "save_audio_in_visqol_format": False,
            },
        }
    },
    "generation-beats": {
        "audio_encoder": encoder_config("beats", "wavcaps"),
        "text_encoder": TEXT_ENCODER_DEFAULT_CONFIG,
        "decoder": decoder_config("beats", CHECKPOINT_DIR / "reconstruction-beats" / "decoder.pt"),
        "clap": CLAP_DEFAULT_CONFIG,
        "generator": {
            "model": {**GENERATOR_MODEL, "latent_dim": 768},
            "schedule": GENERATOR_SCHEDULE,
            "latent_mean": -0.00304481,
            "latent_std": 0.460262,
            "cfg_dropout_prob": 0.1,
            "cfg_scale": 5.0,
        },
        "training": {
            **BASE_GENERATION_TRAINING,
            **BATCH_SETTINGS[32],
            "logger": {"save_dir": LOGS_DIR / "generation-beats"},
            "checkpointing": checkpointing_config(interval=CHECKPOINT_INTERVAL, save_dir=CHECKPOINT_DIR / "generation-beats"),
            "checkpoint": CHECKPOINT_DIR / "generation-beats" / "last.ckpt",
        },
        "evaluation": {
            "checkpoint": CHECKPOINT_DIR / "generation-beats" / "last.ckpt",  # placeholder, point to the checkpoint to evaluate
            "dataset": AUDIOCAPS_EVAL_CONFIG,
            "dataloaders": BATCH_SETTINGS[32]["dataloaders"]["validation"],
            "save": {
                "save_dir": RESULTS_DIR / "generation-beats" / "audiocaps",
                "save_audio_in_visqol_format": False,
            },
        }
    },
    "generation-sao": {
        "audio_encoder": encoder_config("sao", "wavcaps"),
        "text_encoder": TEXT_ENCODER_DEFAULT_CONFIG,
        "decoder": DECODER_DEFAULT_CONFIGS["sao"],
        "clap": CLAP_DEFAULT_CONFIG,
        "generator": {
            "model": {**GENERATOR_MODEL, "latent_dim": 64},
            "schedule": GENERATOR_SCHEDULE,
            "latent_mean": -0.0601637,
            "latent_std": 0.959906,
            "cfg_dropout_prob": 0.1,
            "cfg_scale": 5.0,
        },
        "training": {
            **BASE_GENERATION_TRAINING,
            **BATCH_SETTINGS[32],
            "logger": {"save_dir": LOGS_DIR / "generation-sao"},
            "checkpointing": checkpointing_config(interval=CHECKPOINT_INTERVAL, save_dir=CHECKPOINT_DIR / "generation-sao"),
            "checkpoint": CHECKPOINT_DIR / "generation-sao" / "last.ckpt",
        },
        "evaluation": {
            "checkpoint": CHECKPOINT_DIR / "generation-sao" / "last.ckpt",  # placeholder, point to the checkpoint to evaluate
            "dataset": AUDIOCAPS_EVAL_CONFIG,
            "dataloaders": BATCH_SETTINGS[32]["dataloaders"]["validation"],
            "save": {
                "save_dir": RESULTS_DIR / "generation-sao" / "audiocaps",
                "save_audio_in_visqol_format": False,
            },
        }
    },
    "generation-same": {
        "audio_encoder": encoder_config("same", "wavcaps"),
        "text_encoder": TEXT_ENCODER_DEFAULT_CONFIG,
        "decoder": DECODER_DEFAULT_CONFIGS["same"],
        "clap": CLAP_DEFAULT_CONFIG,
        "generator": {
            "model": {**GENERATOR_MODEL, "latent_dim": 256},
            "schedule": GENERATOR_SCHEDULE,
            "latent_mean": -0.0141349,
            "latent_std": 0.865889,
            "cfg_dropout_prob": 0.1,
            "cfg_scale": 9.0,
        },
        "training": {
            **BASE_GENERATION_TRAINING,
            **BATCH_SETTINGS[32],
            "logger": {"save_dir": LOGS_DIR / "generation-same"},
            "checkpointing": checkpointing_config(interval=CHECKPOINT_INTERVAL, save_dir=CHECKPOINT_DIR / "generation-same"),
            "checkpoint": CHECKPOINT_DIR / "generation-same" / "last.ckpt",
        },
        "evaluation": {
            "checkpoint": CHECKPOINT_DIR / "generation-same" / "last.ckpt",  # placeholder, point to the checkpoint to evaluate
            "dataset": AUDIOCAPS_EVAL_CONFIG,
            "dataloaders": BATCH_SETTINGS[32]["dataloaders"]["validation"],
            "save": {
                "save_dir": RESULTS_DIR / "generation-same" / "audiocaps",
                "save_audio_in_visqol_format": False,
            },
        }
    },
    "generation-audioldm2": {
        "audio_encoder": encoder_config("audioldm2", "wavcaps"),
        "text_encoder": TEXT_ENCODER_DEFAULT_CONFIG,
        "decoder": DECODER_DEFAULT_CONFIGS["audioldm2"],
        "clap": CLAP_DEFAULT_CONFIG,
        "generator": {
            "model": {**GENERATOR_MODEL, "latent_dim": 128},
            "schedule": GENERATOR_SCHEDULE,
            "latent_mean": 0.0571859,
            "latent_std": 0.982315,
            "cfg_dropout_prob": 0.1,
            "cfg_scale": 5.0,
        },
        "training": {
            **BASE_GENERATION_TRAINING,
            **BATCH_SETTINGS[64],
            "logger": {"save_dir": LOGS_DIR / "generation-audioldm2"},
            "checkpointing": checkpointing_config(interval=CHECKPOINT_INTERVAL, save_dir=CHECKPOINT_DIR / "generation-audioldm2"),
            "checkpoint": CHECKPOINT_DIR / "generation-audioldm2" / "last.ckpt",
        },
        "evaluation": {
            "checkpoint": CHECKPOINT_DIR / "generation-audioldm2" / "last.ckpt",  # placeholder, point to the checkpoint to evaluate
            "dataset": AUDIOCAPS_EVAL_CONFIG,
            "dataloaders": BATCH_SETTINGS[64]["dataloaders"]["validation"],
            "save": {
                "save_dir": RESULTS_DIR / "generation-audioldm2" / "audiocaps",
                "save_audio_in_visqol_format": False,
            },
        }
    },
}