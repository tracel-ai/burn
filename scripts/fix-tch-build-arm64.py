#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a helper script to fix burn-tch build issues on Mac M1/M2 machines.

It's a temporary workaround for https://github.com/burn-rs/burn/issues/180 
till tch-rs starts using Torch 2.0 libraries.

This script installs torch via pip3 and creates environment variables in 
.cargo/config.toml for tch-rs to link cc libs properly.


"""

import os
import pathlib


def torch_path():
    import torch
    return pathlib.Path(torch.__file__).parent


def update_toml_config():
    import tomli
    import tomli_w

    cargo_cfg_dir = pathlib.Path(__file__).parent.parent.joinpath(
        ".cargo").resolve()
    cargo_cfg_dir.exists()
    if not cargo_cfg_dir.exists():
        os.makedirs(cargo_cfg_dir)

    toml_file_path = cargo_cfg_dir.joinpath("config.toml")

    # Create toml file if does not exists
    with open(toml_file_path, 'a') as f:
        pass

    with open(toml_file_path, 'rb') as f:
        config = tomli.load(f)

        config["env"] = config.get("env", dict())

        config["env"]["LIBTORCH"] = dict(
            value="{}".format(torch_path()),
            force=True,
        )

        config["env"]["DYLD_LIBRARY_PATH"] = dict(
            value="{}/lib".format(torch_path()),
            force=True,
        )

    with open(toml_file_path, 'wb') as f:
        tomli_w.dump(config, f)


def main():
    print("Installing/Upgrading torch via pip install ...")
    os.system("pip3 install -U torch")
    os.system("pip3 install -U tomli")
    os.system("pip3 install -U tomli-w")

    print("Updating config.toml with torch library paths ... ")
    update_toml_config()


if __name__ == '__main__':
    main()
