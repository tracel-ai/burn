#!/usr/bin/env python3
"""Build an AdamW benchmark and bind its source manifest to the saved executable."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def sources():
    paths = [Path("Cargo.toml"), Path("Cargo.lock"), Path(".cargo/config.toml")]
    for crate in ["burn-optim", "burn-cubecl"]:
        root = Path("crates") / crate
        paths.extend(root.rglob("*.rs"))
        paths.append(root / "Cargo.toml")
    return {str(path): digest(path) for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Saved executable; its manifest is <output>.json")
    parser.add_argument("--features", default="std,cpu,cuda")
    args = parser.parse_args()
    before = sources()
    command = ["cargo", "bench", "-p", "burn-optim", "--bench", "adamw",
               "--no-default-features", "--features", args.features,
               "--no-run", "--locked", "--message-format=json"]
    env = {**os.environ, "RUSTC_WRAPPER": "", "RUSTC_WORKSPACE_WRAPPER": ""}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(args.output) + ".build.log").open("w") as log:
        result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=log)
        artifacts = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        for item in artifacts:
            if item.get("reason") == "compiler-message":
                log.write(item["message"].get("rendered") or item["message"]["message"])
    if result.returncode:
        raise SystemExit(f"Build failed: see {args.output}.build.log")
    if before != sources():
        raise SystemExit("Sources changed during the build; rerun on a stable tree")
    executable, = [item["executable"] for item in artifacts
                   if item.get("reason") == "compiler-artifact"
                   and item["target"]["name"] == "adamw" and item.get("executable")]
    shutil.copy2(executable, args.output)
    manifest = {
        "command": command, "features": args.features, "sources": before,
        "binary_sha256": digest(args.output),
        "rustc": subprocess.check_output([env.get("RUSTC", "rustc"), "-Vv"], text=True).strip(),
        "cargo": subprocess.check_output(["cargo", "-V"], text=True).strip(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "rustflags": env.get("RUSTFLAGS", ""),
        "encoded_rustflags": env.get("CARGO_ENCODED_RUSTFLAGS", ""),
        "wrappers_disabled": True,
    }
    Path(str(args.output) + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
