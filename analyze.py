"""Reproduce the paper's layer-wise probe and latent-distribution analyses.

This script evaluates a trained probe checkpoint or estimates a Student-t
shape parameter at selected encoder stages. It intentionally performs one
deterministic run; candidate-seed search is not part of the public pipeline.
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from distribution import fit_student_t_nu_profile
from utils import build_dataset, load_model, seed_everything


DEFAULT_LAYERS = ("enc_1", "enc_2", "enc_3_0", "enc_3_1", "enc_3_2", "enc_4", "enc_proj")


def collect_activations(network, loader, device, layer_names, max_samples):
    values = {name: [] for name in layer_names}
    hooks = []

    for name in layer_names:
        module = getattr(network, name)

        def capture(_, __, output, layer=name):
            flat = output.detach().float().flatten().cpu()
            remaining = max_samples - sum(item.numel() for item in values[layer])
            if remaining > 0:
                values[layer].append(flat[:remaining])

        hooks.append(module.register_forward_hook(capture))

    with torch.inference_mode():
        for image, _ in tqdm(loader, desc="collect"):
            network(image.to(device))
            if all(sum(item.numel() for item in chunks) >= max_samples for chunks in values.values()):
                break
    for hook in hooks:
        hook.remove()
    return {name: torch.cat(chunks) for name, chunks in values.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--layers", nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--max_samples", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1030)
    parser.add_argument("--output", default="latent_statistics.npz")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    network, config, _ = load_model(args.experiment, args.log_dir, device)
    image_size = int(config["model"].get("img_shape", [256, 256])[0])
    dataset = build_dataset(args.data_root, image_size, training=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    activations = collect_activations(network, loader, device, args.layers, args.max_samples)

    result = {}
    for layer, value in activations.items():
        standardized = (value - value.mean()) / value.std().clamp_min(1e-8)
        nu, _ = fit_student_t_nu_profile(standardized)
        result[f"{layer}_nu"] = np.asarray(float(nu))
        result[f"{layer}_mean"] = np.asarray(float(value.mean()))
        result[f"{layer}_std"] = np.asarray(float(value.std()))
        print(f"{layer}: nu={float(nu):.4f}")
    np.savez(args.output, **result)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
