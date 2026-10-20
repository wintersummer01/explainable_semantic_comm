"""Evaluate reconstruction quality or linear-probe accuracy."""

import argparse

import torch
from tqdm import tqdm

from criterion import PROBE_KEYS
from utils import build_dataset, load_model, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="experiment name, directory, or checkpoint path")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1030)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    network, config, checkpoint = load_model(args.experiment, args.log_dir, device)
    image_size = int(config["model"].get("img_shape", [256, 256])[0])
    dataset = build_dataset(args.data_root, image_size=image_size, training=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    mse_sum = 0.0
    count = 0
    correct = {key: 0 for key in PROBE_KEYS}
    with torch.inference_mode():
        for image, label in tqdm(loader):
            image, label = image.to(device), label.to(device)
            output = network(image)
            count += image.shape[0]
            if "x_hat" in output:
                per_image_mse = (output["x_hat"].clamp(0, 1) - image).square().flatten(1).mean(1)
                mse_sum += float(per_image_mse.sum())
            else:
                for key in PROBE_KEYS:
                    correct[key] += int((output[key].argmax(1) == label).sum())

    print(f"checkpoint: {checkpoint}")
    if mse_sum:
        mse = mse_sum / count
        print(f"MSE: {mse:.8f}")
        print(f"PSNR: {10 * torch.log10(torch.tensor(1.0 / mse)).item():.4f} dB")
    else:
        for key in PROBE_KEYS:
            print(f"{key}: {100.0 * correct[key] / count:.2f}%")


if __name__ == "__main__":
    main()
