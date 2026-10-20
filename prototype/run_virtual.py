"""Transmit one image through the published OFDM chain and an AWGN waveform channel."""

import argparse
import math

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from prototype.jscc_codec import JSCCCodec
from prototype.ofdm import ResourceGrid, OFDMDemodulatorThread, OFDMModulatorThread, OFDMSynchronizerThread
from utils import load_model, seed_everything


def add_awgn(waveform, snr_db, rng):
    power = np.mean(np.abs(waveform) ** 2)
    variance = power / (10 ** (snr_db / 10))
    noise = np.sqrt(variance / 2) * (
        rng.standard_normal(waveform.shape) + 1j * rng.standard_normal(waveform.shape)
    )
    return (waveform + noise).astype(np.complex64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("image")
    parser.add_argument("--output", default="reconstruction.png")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--bandwidth", type=float, choices=(1.4, 3, 5, 10, 15, 20), default=5)
    parser.add_argument("--seed", type=int, default=1030)
    args = parser.parse_args()

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    network, config, _ = load_model(args.experiment, args.log_dir, args.device)
    codec = JSCCCodec(network)
    image_size = int(config["model"].get("img_shape", [256, 256])[0])
    transform = v2.Compose([
        v2.Resize(image_size + 32),
        v2.CenterCrop((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(args.device)
    symbols, latent_shape = codec.encode(image)
    tx_symbols = symbols.squeeze(0).cpu().numpy().astype(np.complex64)

    grid = ResourceGrid(args.bandwidth, N_ports=1, power=1 / 8)
    modulator = OFDMModulatorThread(grid)
    synchronizer = OFDMSynchronizerThread(grid, symbols_per_gop=tx_symbols.size)
    demodulator = OFDMDemodulatorThread(grid, symbols_per_gop=tx_symbols.size)
    received = []
    for frame in modulator.process(tx_symbols):
        noisy_frame = add_awgn(frame, args.snr_db, rng)
        corrected_frame = synchronizer.process(noisy_frame)
        received.append(demodulator.process(corrected_frame))
    rx_symbols = np.concatenate(received)[: tx_symbols.size]

    rx_tensor = torch.from_numpy(rx_symbols).to(args.device).unsqueeze(0)
    reconstruction = codec.decode(rx_tensor, latent_shape).clamp(0, 1)
    mse = (reconstruction - image).square().mean()
    psnr = 10 * torch.log10(mse.reciprocal())
    v2.ToPILImage()(reconstruction.squeeze(0).cpu()).save(args.output)
    print(f"frames: {math.ceil(tx_symbols.size / grid.num_data_symbols)}")
    print(f"PSNR: {float(psnr):.4f} dB")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
