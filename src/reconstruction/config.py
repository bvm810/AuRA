from src.common.constants import *

BASE_RECONSTRUCTION_TRAINING = {
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
            "total_iters": 144_000,
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
        "max_steps": 144_000,
        "num_nodes": 1,
        "devices": 4,
        "precision": "bf16-mixed",
    },
    "validation_losses": {
        "MR-STFT": {
                "weight": 1.0,
                "params": {
                    "fft_sizes": [1024, 2048, 512],
                    "win_lengths": [600, 1250, 350],
                    "hop_sizes": [120, 250, 70],
                }
        },
        "SI-SDR": {"weight": 1.0},
    },
    "additional_logging": {
        "heavy_logging_interval": 14_400,
        "num_heavy_log_batches": 5,                         
        "log_total_grad_norm": True,
        "log_detailed_grad_norm": False,
    },
    "dataset": AUDIOSET_CONFIG
}

CONFIGS = {
    "reconstruction-matpac": {
        "encoder": encoder_config("matpac"),
        "decoder": decoder_config("matpac", None),
        "training": {
            **BASE_RECONSTRUCTION_TRAINING,
            "logger": {"save_dir": LOGS_DIR / "reconstruction-matpac"},
            "checkpointing": checkpointing_config(interval=14_400, save_dir=CHECKPOINT_DIR / "reconstruction-matpac"),
            "checkpoint": CHECKPOINT_DIR / "reconstruction-matpac" / "last.ckpt",
            "dataloaders": loader_config(batch_size=64, num_workers=10),
        },
        "evaluation": {
            "dataset": AUDIOCAPS_EVAL_CONFIG,
            "dataloaders": loader_config(batch_size=64, num_workers=10)["validation"],
            "save": {
                "save_dir": RESULTS_DIR / "reconstruction-matpac" / "audiocaps",
                "save_original_audio": True,
                "save_reconstructed_audio": True,
                "save_audio_in_visqol_format": True,
            },
        }
    },
    "reconstruction-beats": {
        "encoder": encoder_config("beats"),
        "decoder": decoder_config("beats", None),
        "training": {
            **BASE_RECONSTRUCTION_TRAINING,
            "logger": {"save_dir": LOGS_DIR / "reconstruction-beats"},
            "checkpointing": checkpointing_config(interval=14_400, save_dir=CHECKPOINT_DIR / "reconstruction-beats"),
            "checkpoint": CHECKPOINT_DIR / "reconstruction-beats" / "last.ckpt",
            "dataloaders": loader_config(batch_size=64, num_workers=10),
        },
        "evaluation": {
            "dataset": AUDIOCAPS_EVAL_CONFIG,
            "dataloaders": loader_config(batch_size=64, num_workers=10)["validation"],
            "save": {
                "save_dir": RESULTS_DIR / "reconstruction-beats" / "audiocaps",
                "save_original_audio": True,
                "save_reconstructed_audio": True,
                "save_audio_in_visqol_format": True,
            },
        }
    },
}