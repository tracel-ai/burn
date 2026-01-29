#!/usr/bin/env python3
"""LPIPS comparison script for Burn vs PyTorch.

Usage:
    # Generate weights first
    python compare.py --generate-weights

    # Run comparison
    python compare.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import lpips


def load_raw_image(path: str) -> torch.Tensor:
    """Load raw image (64x64x3 f32) as tensor."""
    data = np.fromfile(path, dtype=np.float32).reshape(64, 64, 3)
    return torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0)


def generate_weights():
    """Generate pretrained weights for Burn."""
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    print("Generating VGG weights...")
    loss_vgg = lpips.LPIPS(net='vgg')
    torch.save(loss_vgg.state_dict(), weights_dir / "lpips_vgg.pt")
    print(f"  Saved: {weights_dir / 'lpips_vgg.pt'}")

    print("Generating AlexNet weights...")
    loss_alex = lpips.LPIPS(net='alex')
    torch.save(loss_alex.state_dict(), weights_dir / "lpips_alex.pt")
    print(f"  Saved: {weights_dir / 'lpips_alex.pt'}")

    print("\nDone! Now run:")
    print("  cargo run -p lpips-test")
    print("  python compare.py")


def run_tests(loss_fn, net_name: str):
    """Run LPIPS tests."""
    img_h = load_raw_image("img/test_img_horizontal.raw")
    img_v = load_raw_image("img/test_img_vertical.raw")
    img_d = load_raw_image("img/test_img_diagonal.raw")

    with torch.no_grad():
        # Test 1: zeros vs ones
        zeros = torch.zeros(1, 3, 64, 64)
        ones = torch.ones(1, 3, 64, 64)
        loss1 = loss_fn(zeros, ones, normalize=True)
        print(f"  zeros vs ones:      {loss1.item():.6f}")

        # Test 2: horizontal vs vertical
        loss2 = loss_fn(img_h, img_v, normalize=True)
        print(f"  horizontal vs vertical: {loss2.item():.6f}")

        # Test 3: horizontal vs diagonal
        loss3 = loss_fn(img_h, img_d, normalize=True)
        print(f"  horizontal vs diagonal: {loss3.item():.6f}")


def main():
    parser = argparse.ArgumentParser(description="LPIPS comparison")
    parser.add_argument("--generate-weights", action="store_true", help="Generate weights")
    args = parser.parse_args()

    if args.generate_weights:
        generate_weights()
        return

    print("=== LPIPS Test (PyTorch) ===\n")

    # VGG
    print("[VGG]")
    loss_vgg = lpips.LPIPS(net='vgg')
    run_tests(loss_vgg, "VGG")

    # AlexNet
    print("\n[AlexNet]")
    loss_alex = lpips.LPIPS(net='alex')
    run_tests(loss_alex, "AlexNet")

    print("\nCompare with: cargo run -p lpips-test")


if __name__ == "__main__":
    main()
