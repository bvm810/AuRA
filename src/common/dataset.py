import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        preprocessing_params: Dict[str, Any],
        csv_name: str = "metadata.csv",
        audio_dir_name: str = "audio",
        audio_column: str = "path",
        split_column: str = "split",
        split: Optional[str] = None,
        deterministic_crop: bool = False,
        return_path: bool = False,
    ):
        self.root_dir = root_dir
        self.audio_column = audio_column
        self.split_column = split_column
        self.csv_path = os.path.join(root_dir, csv_name)
        self.audio_dir = os.path.join(root_dir, audio_dir_name)

        self.metadata = pd.read_csv(self.csv_path)
        for column in self._required_columns():
            if column not in self.metadata.columns:
                raise KeyError(f"Column '{column}' not found in {self.csv_path}. Available columns: {list(self.metadata.columns)}")

        self.split = None
        if split is not None:
            self.set_split(split)

        self.preprocessing_params = preprocessing_params
        self.deterministic_crop = deterministic_crop
        self.return_path = return_path
        self._refresh_paths()
        self._resamplers = {}

    # ---------------------------------------------------------------- helpers

    def _required_columns(self) -> List[str]:
        return [self.audio_column]

    def _refresh_paths(self) -> None:
        """Cache the per-row fields currently held in self.metadata."""
        self.paths = self.metadata[self.audio_column].tolist()

    def resolve_path(self, rel_path: str) -> str:
        """CSV paths are relative to root_dir; fall back to root_dir/audio_dir."""
        if os.path.isabs(rel_path):
            return rel_path
        candidate = os.path.join(self.root_dir, rel_path)
        if os.path.exists(candidate):
            return candidate
        return os.path.join(self.audio_dir, rel_path)

    def set_split(self, split: str) -> None:
        if self.split_column not in self.metadata.columns:
            raise KeyError(f"Column '{self.split_column}' not found in {self.csv_path}; cannot filter by split.")
        self.split = split
        self.metadata = self.metadata[self.metadata[self.split_column] == split]
        self.metadata = self.metadata.reset_index(drop=True)
        self._refresh_paths()

    # ------------------------------------------------------------ dataset api

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        rel_path = self.paths[idx]
        full_path = self.resolve_path(rel_path)
        try:
            audio = self.load_audio(full_path)
        except Exception as e:
            print(f"Error processing path {full_path} (idx {idx}): {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
        return self._make_item(idx, audio.squeeze(0), rel_path)

    def _make_item(self, idx: int, audio: torch.Tensor, rel_path: str) -> Union[torch.Tensor, Tuple[torch.Tensor, str]]:
        if self.return_path:
            return audio, rel_path
        return audio

    # ------------------------------------------------------------------ audio

    def load_audio(self, path: str) -> torch.Tensor:
        target_len = self.preprocessing_params["segment_length"]
        info = torchaudio.info(path)
        crop_size = int(target_len * (info.sample_rate / self.preprocessing_params["sample_rate"]))
        if self.deterministic_crop:
            start = 0
        else:
            start = random.randint(0, max(info.num_frames - crop_size, 0))
        audio, sr = torchaudio.load(path, frame_offset=start, num_frames=crop_size)
        if audio.shape[0] != 1:
            audio = audio.mean(dim=0).reshape(1, -1)
        if sr != self.preprocessing_params["sample_rate"]:
            audio = self._resample(audio, sr)
        return self._fix_length(audio, target_len)

    def _resample(self, x: torch.Tensor, sr: int) -> torch.Tensor:
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.preprocessing_params["sample_rate"],
                rolloff=0.99,
                resampling_method="sinc_interp_kaiser",
                lowpass_filter_width=64,
            ).to(x.device)
        return self._resamplers[sr](x)

    def _fix_length(self, audio: torch.Tensor, target_len: int) -> torch.Tensor:
        n = audio.shape[-1]
        if n > target_len:  # truncate
            audio = audio[..., :target_len]
        elif n < target_len:  # right-pad with zeros
            audio = torch.nn.functional.pad(audio, (0, target_len - n))
        return audio


class CaptioningDataset(AudioDataset):
    def __init__(self, caption_column: str = "caption", **kwargs):
        self.caption_column = caption_column  # set first: the parent's __init__ validates columns and calls _refresh_paths
        super().__init__(**kwargs)

    def _required_columns(self) -> List[str]:
        return super()._required_columns() + [self.caption_column]

    def _refresh_paths(self) -> None:
        super()._refresh_paths()
        self.captions = self.metadata[self.caption_column].tolist()

    def _make_item(self, idx: int, audio: torch.Tensor, rel_path: str) -> Union[Tuple[torch.Tensor, str, str], Tuple[torch.Tensor, str]]:
        caption = self.captions[idx]
        if self.return_path:
            return audio, caption, rel_path
        return audio, caption