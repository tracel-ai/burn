#!/usr/bin/env python3
"""Prepare the local CubeCL getter dependency without modifying Cargo's source cache."""

import argparse
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cubecl_checkout", type=Path, help="Existing CubeCL git checkout containing the pinned revision")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    destination = root / "target/adamw-foreach-cubecl"
    revision = "d88b0febb50e7b83a8548125f6a701c12f11a2ba"
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite {destination}; use an absent destination.")
    subprocess.run(["git", "-C", str(args.cubecl_checkout), "cat-file", "-e", revision + "^{commit}"], check=True)
    destination.mkdir(parents=True)
    archive_path = root / "target/adamw-foreach-cubecl.tar"
    with archive_path.open("wb") as archive:
        subprocess.run(["git", "-C", str(args.cubecl_checkout), "archive", revision], stdout=archive, check=True)
    subprocess.run(["tar", "-xf", str(archive_path), "-C", str(destination)], check=True)
    patch = root / "docs/optim-adamw/prerequisites/cubecl-execution-stream.patch"
    subprocess.run(["git", "apply", "--directory=target/adamw-foreach-cubecl", str(patch)], cwd=root, check=True)
    config = ['[patch."https://github.com/tracel-ai/cubecl"]']
    for manifest in sorted((destination / "crates").glob("*/Cargo.toml")):
        text = manifest.read_text()
        name = re.search(r'^name\s*=\s*"([^"]+)"', text, re.M)
        if name:
            # Without an explicit workspace root Cargo may infer the enclosing Burn workspace.
            manifest.write_text(text.replace("[package]", '[package]\nworkspace = "../.."', 1))
            config.append(f"{name[1]} = {{ path = {json.dumps(str(manifest.parent))} }}")
    config_path = root / "target/adamw-foreach-local.toml"
    config_path.write_text("\n".join(config) + "\n")
    print(config_path)
    print("The Burn candidate patch and production dependency pins have not been applied.")


if __name__ == "__main__":
    main()
