from __future__ import annotations

import argparse
import time

import torch

from klh.attribution import KLHessianAttributor
from klh.config import AttributionConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure attribution wall-clock time")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10))

        def forward(self, x):
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Tiny().to(device).eval()
    image = torch.rand(1, 3, 32, 32, device=device)
    masks = torch.zeros(4, 32, 32, device=device)
    masks[0, :16, :16] = 1
    masks[1, :16, 16:] = 1
    masks[2, 16:, :16] = 1
    masks[3, 16:, 16:] = 1
    attr = KLHessianAttributor(model, AttributionConfig())

    for _ in range(2):
        attr.attribute(image, masks)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iterations):
        attr.attribute(image, masks)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"mean_seconds_per_image={elapsed / args.iterations:.6f}")


if __name__ == "__main__":
    main()
