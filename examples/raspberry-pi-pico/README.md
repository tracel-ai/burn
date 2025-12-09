# Running Onnx Inference on the Raspberry Pi Pico 2
This example shows how to run an inference on a no_std, no atomic pointer, and no heap environment.

## Setup
1. Install Raspberry Pi Pico 2 target `rustup target add thumbv8m.main-none-eabihf`

2. Install `elf2flash` with `cargo install elf2flash`. Optionally install [`probe-rs`](https://probe.rs/docs/getting-started/installation/) instead.

If you have opted to use probe-rs, have a [compatible probe](https://probe.rs/docs/getting-started/probe-setup/) to flash to the raspberry pi pico. Modify the `.cargo/config.toml` and uncomment the runner to use `probe-rs`.

## Running
Run as usual with `cargo run`

## Project Structure
The project is structured as follows
```
raspberry-pi-pico
├── Cargo.lock
├── Cargo.toml
├── README.md
├── build.rs
├── memory.x
├── src
│   ├── bin
│   │   └── main.rs
│   ├── lib.rs
│   └── model
│       ├── mod.rs
│       └── sine.onnx
└── tensorflow
    ├── requirements.txt
    └── train.py
```
Everything is standard with any other cargo project except for the `memory.x`, the `model` directory, and the `tensorflow` directory.

The `memory.x` file contains the memory layout of the chip.

The `tensorflow` directory contains a python script which generates the onnx model using tensorflow, using the requirements from `requirements.txt`.
The onnx model will be outputted to `src/model/sine.onnx`. The `build.rs` script will generate a rust file which takes in the `sine.onnx` file and generates an import, which gets exposed in `mod.rs`.
