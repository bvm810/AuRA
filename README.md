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
- [ ] Clean neural network modules file
- [ ] Clean configs / constants
- [ ] Clean dataloader
- [ ] Clean Lightning Module
- [ ] Clean eval scripts 
- [ ] Upload checkpoints to TTA models

### Setup

**TODO -- Setup**
- [ ] Add `environment.yaml` file to the repo
- [ ] Write dependency install instructions
- [ ] Add links to third-party checkpoints used in the paper
- [ ] Add link to AuRA checkpoints and download instructions
- [ ] State VRAM requirements and GPUs used to train the models

### Usage

**TODO -- Usage**
- [ ] Write *reconstruction* subsection, with instructions on how to run the models for encoding and decoding a signal
- [ ] Write *generation* subsection with insturctions on how to run the models for TTA
- [ ] Write instruction to recalculate paper metrics

### Data

**TODO -- Datasets**
- [ ] Add data cleaning scripts for WavCaps and AudioCaps
- [ ] Write instructions on how to clean data

### Acknowledgements

This paper is built on [RAE](https://github.com/bytetriper/RAE) and [MATPAC++](https://github.com/aurianworld/matpac/tree/main) code. We would like to thank the authors for their contributions.
