# Burn Torch Backend

[Burn](https://github.com/tracel-ai/burn) Torch backend

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-tch.svg)](https://crates.io/crates/burn-tch)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn-tch/blob/master/README.md)

This crate provides a Torch backend for [Burn](https://github.com/tracel-ai/burn) utilizing the
[`tch-rs`](https://github.com/LaurentMazare/tch-rs) crate, which offers a Rust interface to the
[PyTorch](https://pytorch.org/) C++ API.

The backend supports CPU (multithreaded), [CUDA](https://pytorch.org/docs/stable/notes/cuda.html)
(multiple GPUs), and [MPS](https://pytorch.org/docs/stable/notes/mps.html) devices (MacOS).

## Installation

[`tch-rs`](https://github.com/LaurentMazare/tch-rs) requires the C++ PyTorch libray (LibTorch) to be
available on your system. It is recommended to manually download the LibTorch distribution (v2.1.0
as required by `tch-rs`) as per the instructions below, except for [MPS](#mps).

| Compute Platform          |              CPU               | GPU | Linux | MacOS | Windows | Android | iOS | WASM |
| :------------------------ | :----------------------------: | :-: | :---: | :---: | :-----: | :-----: | :-: | :--: |
| [CPU](#cpu)               |              Yes               | No  |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
| [CUDA](#cuda)             | Yes <sup>[[1]](#cpu-sup)</sup> | Yes |  Yes  |  No   |   Yes   |   No    | No  |  No  |
| [Metal (MPS)](#metal-mps) |               No               | Yes |  No   |  Yes  |   No    |   No    | No  |  No  |
| Vulkan                    |              Yes               | Yes |  Yes  |  Yes  |   Yes   |   Yes   | No  |  No  |

<sup><a id="cpu-sup">[1]</a> The LibTorch CUDA distribution also comes with CPU support.</sup>

Once your installation is complete, you should be able to build/run your project. You can also
validate your installation by running the simple example below.

```shell
cargo run --example simple-add --release
```

### CPU

<details open>
<summary><strong>🐧 Linux</strong></summary>

First, download the LibTorch CPU distribution.

```shell
wget -O libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch.zip
```

Then, point to that installation using the `LIBTORCH` and `LD_LIBRARY_PATH` environment variables
before building `burn-tch` or a crate which depends on it.

```shell
export LIBTORCH=/absolute/path/to/libtorch/
export LD_LIBRARY_PATH=/absolute/path/to/libtorch/lib:$LD_LIBRARY_PATH
```

</details><br>

<details>
<summary><strong>🍎 Mac</strong></summary>

First, download the LibTorch CPU distribution.

```shell
wget -O libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.0.zip
unzip libtorch.zip
```

Then, point to that installation using the `LIBTORCH` and `DYLD_LIBRARY_PATH` environment variables
before building `burn-tch` or a crate which depends on it.

```shell
export LIBTORCH=/absolute/path/to/libtorch/
export DYLD_LIBRARY_PATH=/absolute/path/to/libtorch/lib:$DYLD_LIBRARY_PATH
```

</details><br>

<details>
<summary><strong>🪟 Windows</strong></summary>

First, download the LibTorch CPU distribution.

```powershell
wget https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.1.0%2Bcpu.zip -OutFile libtorch.zip
Expand-Archive libtorch.zip
```

Then, set the `LIBTORCH` environment variable and append the library to your path as with the
PowerShell commands below before building `burn-tch` or a crate which depends on it.

```powershell
$Env:LIBTORCH = "/absolute/path/to/libtorch/"
$Env:Path += ";/absolute/path/to/libtorch/"
```

</details><br>

### CUDA

LibTorch 2.1.0 currently supports CUDA
[12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive) or
[11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive). Just make sure you have the
correct toolkit installed, otherwise a number of issues could arise.

**CUDA 12.1**

<details open>
<summary><strong>🐧 Linux</strong></summary>

First, download the LibTorch CUDA 12.1 distribution.

```shell
wget -O libtorch.zip https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
unzip libtorch.zip
```

Then, point to that installation using the `LIBTORCH` and `LD_LIBRARY_PATH` environment variables
before building `burn-tch` or a crate which depends on it.

```shell
export LIBTORCH=/absolute/path/to/libtorch/
export LD_LIBRARY_PATH=/absolute/path/to/libtorch/lib:$LD_LIBRARY_PATH
```

**Note:** make sure your CUDA installation is in your `PATH` and `LD_LIBRARY_PATH`.

</details><br>

<details>
<summary><strong>🪟 Windows</strong></summary>

First, download the LibTorch CUDA 12.1 distribution.

```powershell
wget https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.1.0%2Bcu121.zip -OutFile libtorch.zip
Expand-Archive libtorch.zip
```

Then, set the `LIBTORCH` environment variable and append the library to your path as with the
PowerShell commands below before building `burn-tch` or a crate which depends on it.

```powershell
$Env:LIBTORCH = "/absolute/path/to/libtorch/"
$Env:Path += ";/absolute/path/to/libtorch/"
```

</details><br>

**CUDA 11.8**

<details open>
<summary><strong>🐧 Linux</strong></summary>

First, download the LibTorch CUDA 11.8 distribution.

```shell
wget -O libtorch.zip https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
unzip libtorch.zip
```

Then, point to that installation using the `LIBTORCH` and `LD_LIBRARY_PATH` environment variables
before building `burn-tch` or a crate which depends on it.

```shell
export LIBTORCH=/absolute/path/to/libtorch/
export LD_LIBRARY_PATH=/absolute/path/to/libtorch/lib:$LD_LIBRARY_PATH
```

**Note:** make sure your CUDA installation is in your `PATH` and `LD_LIBRARY_PATH`.

</details><br>

<details>
<summary><strong>🪟 Windows</strong></summary>

First, download the LibTorch CUDA 11.8 distribution.

```powershell
wget https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.1.0%2Bcu118.zip -OutFile libtorch.zip
Expand-Archive libtorch.zip
```

Then, set the `LIBTORCH` environment variable and append the library to your path as with the
PowerShell commands below before building `burn-tch` or a crate which depends on it.

```powershell
$Env:LIBTORCH = "/absolute/path/to/libtorch/"
$Env:Path += ";/absolute/path/to/libtorch/"
```

</details><br>

### Metal (MPS)

There is no official LibTorch distribution with MPS support at this time, so the easiest alternative
is to use a PyTorch installation. This requires a Python installation.

_Note: MPS acceleration is available on MacOS 12.3+._

```shell
pip install torch==2.1.0
export LIBTORCH_USE_PYTORCH=1
export DYLD_LIBRARY_PATH=/path/to/pytorch/lib:$DYLD_LIBRARY_PATH
```

## Example Usage

For a simple example, check out [simple-add.rs](examples/simple-add.rs). It automatically selects
the device for your installation and performs a simple elementwise addition.

For a more complete example using the `tch` backend, take a loot at the
[Burn mnist example](https://github.com/tracel-ai/burn/tree/main/examples/mnist).
