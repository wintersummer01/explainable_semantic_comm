from __future__ import annotations

import os
from pathlib import Path

import torch
import yaml


def find_checkpoint(experiment, log_dir=None):
    candidate = Path(experiment).expanduser()
    if candidate.is_file():
        return candidate

    root = Path(log_dir or os.environ.get("JSCC_LOG_DIR", "logs"))
    experiment_dir = candidate if candidate.is_dir() else root / experiment
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")

    for pattern in ("best_ckpt*.pth.tar", "latest_ckpt*.pth.tar"):
        matches = sorted(experiment_dir.glob(pattern))
        if matches:
            return matches[-1]
    raise FileNotFoundError(f"No best/latest checkpoint found in {experiment_dir}")


# Backward-compatible name used by the adaptation model constructors.
def get_ckpt(exp_name, log_dir=None):
    return str(find_checkpoint(exp_name, log_dir))


def load_experiment(experiment, log_dir=None):
    checkpoint = find_checkpoint(experiment, log_dir)
    experiment_dir = checkpoint.parent
    config_path = experiment_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing experiment config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config, checkpoint


def load_model(experiment, log_dir=None, device="cpu"):
    import model as model_zoo

    config, checkpoint_path = load_experiment(experiment, log_dir)
    model_config = dict(config["model"])
    model_type = model_config.pop("type")
    network = getattr(model_zoo, model_type)(**model_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    network.load_state_dict(checkpoint["state_dict"], strict=False)
    network.to(device).eval()
    return network, config, checkpoint_path
