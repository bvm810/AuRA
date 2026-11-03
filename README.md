## Exploring Representation Autoencoders for Text-to-Audio Generation (AuRA) <br><sub>Official PyTorch Implementation</sub>

[Demo Page](https://aura-tta.onrender.com)

This repository contains the official implementation of the "Exploring Representation Autoencoders for Text-to-Audio Generation" paper, submitted to ICASSP 2027. This page is still under construction, and will be updated as the code is refactored for publication.

Our paper is, to the best of our knowledge, the first application of audio [representation autoencoders](https://github.com/bytetriper/RAE) (AuRA). In addition to providing trained decoders for [MATPAC++](https://github.com/aurianworld/matpac/tree/main) and [BEATs](https://github.com/microsoft/unilm/tree/master/beats), we also evaluate the two AuRAs against VAEs commonly used in text-to-audio generation using a common generative backbone. A link to the paper with more details will be provided upon acceptance.

**TODO -- Reconstruction Code**
- [x] Clean neural network modules file
  - [x] DiT blocks
  - [x] Decoder
  - [x] Schedules
- [x] Clean configs
- [x] Clean dataloader
- [x] Clean Lightning Module
- [x] Clean factory
- [x] Inference script
- [ ] Test training
- [ ] Test inference
- [ ] Upload checkpoints to MATPAC++ and BEATs decoders

**TODO -- Generation Code**
- [x] Clean neural network modules file
- [x] Clean configs / constants
- [x] Clean dataloader
- [x] Clean Lightning Module
- [x] Clean inference
- [x] Clean scaling curve inference
- [ ] Test training
- [ ] Test inference
- [ ] Upload checkpoints to TTA models

### Setup

Run the commands below to install and activate the AuRA conda environment and then install the MATPAC++ inference package without dependencies.

```
conda env create -f environment.yaml
conda activate aura
pip install --no-deps "matpac @ git+https://github.com/aurianworld/matpac.git#subdirectory=inference_matpac"
```

After this, make sure to have the SSL encoder and VAE checkpoints downloaded:

- **MATPAC++**: `matpac_plus_6s_2048_enconly.pt` from the [MATPAC releases](https://github.com/aurianworld/matpac/releases), saved as `checkpoints/matpac/matpac_pretrained_encoder.pt`.
- **BEATs**: `BEATs_iter3.pt` from [microsoft/unilm](https://github.com/microsoft/unilm/tree/master/beats), saved as `checkpoints/beats/beats_iter3_pretrained_encoder.pt`.
- **Stable Audio Open VAE** (SA-VAE): autoencoder of [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0), saved as `checkpoints/sao/sao-vae.json` (`vae_model_config.json` in HF) and `checkpoints/sao/sao-vae.ckpt` (`vae_model.ckpt` in HF, weights under `state_dict`).
- **SAME-S**: [stabilityai/SAME-S](https://huggingface.co/stabilityai/SAME-S), saved as `checkpoints/same/same.json` (`model_config.json` in HF) and `checkpoints/same/same.ckpt` (`model.safetensors` in HF, converted to a `torch` checkpoint).
- **AudioLDM2 VAE and HiFi-GAN vocoder**: the `vae/` and `vocoder/` folders of [cvssp/audioldm2](https://huggingface.co/cvssp/audioldm2), e.g. `hf download cvssp/audioldm2 --include "vae/*" "vocoder/*" --local-dir checkpoints/audioldm2`.

[T5](https://huggingface.co/google-t5/t5-base) and [CLAP](https://huggingface.co/laion/clap-htsat-unfused) checkpoints are downloaded automatically through HF. Finally, download the AuRA decoder checkpoints and generator checkpoints [here](#). Decoders and generators were trained using 4 H100 GPUs with 80Gb VRAM. 

### Usage

All commands are run from the repository root. Each run is selected by a configuration ID from `src/reconstruction/config.py` or `src/generation/config.py`, which also hold the data, hardware and evaluation settings. Training uses 4 GPUs with DDP by default, saves checkpoints to `checkpoints/<config_id>/` and TensorBoard logs to `logs/<config_id>/`; inference writes to `results/<config_id>/`. Add `--resume` to a training command to continue from `checkpoints/<config_id>/last.ckpt`. Inference evaluates the checkpoint set in the config's `evaluation.checkpoint`, unless one is given with `--checkpoint`.

#### Reconstruction

Configs: `reconstruction-matpac`, `reconstruction-beats`.

Train a decoder on AudioSet:

```bash
python -m src.reconstruction.train --config_id reconstruction-matpac
```

Evaluate it on the AudioCaps test set:

```bash
python -m src.reconstruction.inference --config_id reconstruction-matpac --checkpoint checkpoints/reconstruction-matpac/last.ckpt
```

This writes per-clip MR-STFT scores to `results.csv`, their means and standard errors to `summary.csv`, and the original and reconstructed clips as 48 kHz WAV files inside the results directory for ViSQOL. To evaluate on WavCaps-test instead, set the config's `evaluation.dataset` to `WAVCAPS_EVAL_CONFIG` and change `evaluation.save.save_dir` accordingly.

Export the EMA weights of the decoder for use in generation:

```bash
python -m src.utils.export_lightning_ckpt --config_id reconstruction-matpac --out_path checkpoints/reconstruction-matpac/decoder.pt
```

#### Generation

Configs: `generation-matpac` and `generation-beats` (AuRAs, using the decoders exported above), and `generation-sao`, `generation-same` and `generation-audioldm2` (baselines).

Train a generator on WavCaps and the AudioCaps train split:

```bash
python -m src.generation.train --config_id generation-matpac
```

Generate audio for the AudioCaps test captions and evaluate it:

```bash
python -m src.generation.inference --config_id generation-matpac --checkpoint checkpoints/generation-matpac/last.ckpt
```

Use `--cfg_scale` to override the config's guidance scale. Outputs go to `results/<config_id>/audiocaps/cfg-scale-<scale>/`: the generated clips as 16 kHz FLAC files in `generated-audio/`, the CLAP score of each caption in `results.csv`, and FAD and the mean CLAP score with its standard error in `summary.csv`. To evaluate on WavCaps-test instead, set the config's `evaluation.dataset` to `WAVCAPS_EVAL_CONFIG` and change `evaluation.save.save_dir` accordingly.

### Data

**TODO -- Datasets**
- [ ] Add data cleaning scripts for WavCaps and AudioCaps
- [ ] Write instructions on how to clean data

### Acknowledgements

This paper is built on [RAE](https://github.com/bytetriper/RAE) and [MATPAC++](https://github.com/aurianworld/matpac/tree/main) code. We would like to thank the authors for their contributions.
