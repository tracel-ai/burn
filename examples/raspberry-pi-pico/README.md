# Running Onnx Inference on the Raspberry Pi Pico
This example shows how to run an inference on a no_std, no atomic pointer, and no heap environment.

## Setup
1. Install raspberry pi pico target `rustup target add thumbv6m-none-eabi`

2. Install [`probe-rs`](https://probe.rs/docs/getting-started/installation/). This is optional, install `elf2uf2-rs` to use the usb boot with `cargo install elf2uf2-rs`.

3. Have a [compatible probe](https://probe.rs/docs/getting-started/probe-setup/) to flash to the raspberry pi pico. This is optional, alternatively, modify `.cargo/config.toml` and uncomment the runner to use `elf2uf2-rs`.

If you are using `elfuf2-rs` logging will not go to your serial port, add logging by using `embassy-usb`.

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
