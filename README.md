# Explainable Semantic Communication

Minimal research code for layer-wise analysis and selective adaptation of
neural joint source-channel coding (JSCC) encoders. The repository contains
the Swin Transformer backbone used in the paper, a stage-matched CNN baseline,
linear classification probes, selective fine-tuning models, and a compact
OFDM simulation extracted from the accompanying prototype.

## What is included

- `model/deep_JSCC.py`: Swin and CNN JSCC codecs.
- `model/class_probing.py`: frozen-encoder linear probes.
- `model/split*.py`: 13-block Swin/CNN representations and sliding-window
  encoder fine-tuning.
- `model/freezer*.py`: inner/outer selective adaptation under channel shift.
- `analyze.py`: layer-wise latent collection and Student-t shape estimation.
- `prototype/`: virtual OFDM transmission and optional USRP building blocks.
- `configs/`: runnable examples for base training, probing, and adaptation.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The USRP helper additionally requires the platform-specific UHD Python
bindings. UHD is not needed for model training, evaluation, analysis, or the
virtual OFDM example.

To run the included smoke tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Dataset layout

The loaders use the standard `torchvision.datasets.ImageFolder` layout:

```text
data/imagenette2/
├── train/
│   ├── class_0/
│   └── ...
└── val/
    ├── class_0/
    └── ...
```

ImageNette is the default in the example configurations, but any RGB image
dataset with this layout can be used. Edit `dataset.train_root` and
`dataset.valid_root`, or override them on the command line.

## Base models

```bash
python train.py --config_file configs/swin.yaml
python train.py --config_file configs/cnn.yaml
```

Any YAML value can be overridden with a dotted key:

```bash
python train.py --config_file configs/swin.yaml \
  --experiment_name swin_cbr1o24 \
  --model.spp 0.125 \
  --dataset.train_root /path/to/train \
  --dataset.valid_root /path/to/val
```

Checkpoints and a plain YAML copy of the effective configuration are written
to `logs/<experiment_name>/`.

## Layer-wise classification probes

Set `model.exp` in the probe configuration to the corresponding pretrained
base experiment, then run:

```bash
python train.py --config_file configs/probe_swin.yaml
python train.py --config_file configs/probe_cnn.yaml
python evaluate.py logs/probe_swin --data_root data/imagenette2/val
```

Only the linear heads are updated; the encoder stages remain frozen.

## Selective adaptation

The rate-shift experiment first trains the block-split model at the source
condition and then fine-tunes a consecutive four-block encoder window plus the
decoder at the target condition:

```bash
python train.py --config_file configs/split_swin.yaml
python train.py --config_file configs/rate_swin.yaml --model.start_idx 0

python train.py --config_file configs/split_cnn.yaml
python train.py --config_file configs/rate_cnn.yaml --model.start_idx 0
```

Use `--model.start_idx 0` through `9` to cover all four-block windows. This is
an explicit experimental sweep, not an outcome-driven seed search.

For the channel-distribution shift, use:

```bash
python train.py --config_file configs/channel_swin.yaml \
  --model.num_tune 3 --model.tune_outside false
python train.py --config_file configs/channel_cnn.yaml \
  --model.num_tune 3 --model.tune_outside false
```

Set `model.tune_outside: true` to update the earlier-side encoder stages.

## Reconstruction evaluation and latent analysis

```bash
python evaluate.py logs/swin_awgn10_cbr1o12 \
  --data_root data/imagenette2/val

python analyze.py logs/swin_awgn10_cbr1o12 \
  --data_root data/imagenette2/val \
  --output latent_statistics.npz
```

`evaluate.py` reports PSNR for codecs and per-stage accuracy for probe models.
`analyze.py` saves the fitted Student-t `nu`, mean, and standard deviation for
each encoder stage.

## Virtual OFDM prototype

The prototype transmits one model latent through the LTE-like OFDM resource
grid and an AWGN waveform channel, then reconstructs the image:

```bash
python -m prototype.run_virtual logs/swin_awgn10_cbr1o12 input.png \
  --snr_db 20 --output reconstruction.png
```

The reusable USRP transmit/receive classes are in `prototype/usrp/usrp.py`.
Hardware frequency, gains, addresses, and regulatory constraints are
deployment-specific and therefore are not hardcoded into an executable script.

## Checkpoints

Pretrained weights are not stored in Git. A checkpoint directory must contain:

```text
logs/<experiment>/
├── config.yaml
└── best_ckpt.pth.tar   # or latest_ckpt.pth.tar
```

Adaptation configurations accept either such an experiment directory or a
name resolved relative to `logs/`. The root can also be changed with the
`JSCC_LOG_DIR` environment variable.
