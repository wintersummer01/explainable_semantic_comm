"""Train a codec, linear probes, or one of the selective adaptation models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

import model as model_zoo
from criterion import SemComLoss
from utils.config import apply_overrides, load_config
from utils.data import build_dataloaders
from utils.seed import seed_everything


def move_batch(batch, device):
    if isinstance(batch, (tuple, list)):
        return tuple(item.to(device, non_blocking=True) for item in batch)
    return batch.to(device, non_blocking=True)


def run_epoch(network, loader, criterion, device, optimizer=None, clip_max_norm=None):
    training = optimizer is not None
    network.train(training)
    totals = {}
    examples = 0
    progress = tqdm(loader, desc="train" if training else "valid", leave=False)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in progress:
            batch = move_batch(batch, device)
            image = batch[0] if isinstance(batch, (tuple, list)) else batch
            if training:
                optimizer.zero_grad(set_to_none=True)
            metrics = criterion(network(image), batch)
            if training:
                metrics["loss"].backward()
                if clip_max_norm:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), clip_max_norm)
                optimizer.step()
            batch_size = image.shape[0]
            examples += batch_size
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
            progress.set_postfix(loss=f"{totals['loss'] / examples:.4f}")
    return {name: value / examples for name, value in totals.items()}


def save_checkpoint(path, epoch, network, optimizer, scheduler, metrics):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--experiment_name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args, overrides = parser.parse_known_args()

    config = apply_overrides(load_config(args.config_file), overrides)
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    seed_everything(config.get("seed"))
    device = torch.device(args.device)

    model_config = dict(config["model"])
    model_type = model_config.pop("type")
    network = getattr(model_zoo, model_type)(**model_config).to(device)
    criterion = SemComLoss(config.get("loss_type", "MSE"))
    train_loader, valid_loader = build_dataloaders(**config["dataset"])

    optimizer_config = config["optimizer"]
    optimizer = torch.optim.Adam(
        (parameter for parameter in network.parameters() if parameter.requires_grad),
        lr=float(optimizer_config["learning_rate"]),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=optimizer_config.get("lr_milestone_epoch", []),
        gamma=float(optimizer_config.get("lr_gamma", 0.1)),
    )

    output_dir = Path(config.get("logging", {}).get("save_dir", "logs")) / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    best_loss = float("inf")
    history_path = output_dir / "metrics.jsonl"
    for epoch in range(int(config["epochs"])):
        train_metrics = run_epoch(
            network,
            train_loader,
            criterion,
            device,
            optimizer,
            config.get("clip_max_norm"),
        )
        valid_metrics = run_epoch(network, valid_loader, criterion, device)
        scheduler.step()
        record = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        save_checkpoint(output_dir / "latest_ckpt.pth.tar", epoch, network, optimizer, scheduler, valid_metrics)
        if valid_metrics["loss"] < best_loss:
            best_loss = valid_metrics["loss"]
            save_checkpoint(output_dir / "best_ckpt.pth.tar", epoch, network, optimizer, scheduler, valid_metrics)
        print(f"epoch={epoch} train={train_metrics} valid={valid_metrics}")


if __name__ == "__main__":
    main()
