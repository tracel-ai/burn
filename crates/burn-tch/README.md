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

[`tch-rs`](https://github.com/LaurentMazare/tch-rs) requires the C++ PyTorch library (LibTorch) to
be available on your system.

By default, the CPU distribution is installed for LibTorch v2.2.0 as required by `tch-rs`.

<details>
<summary><strong>CUDA</strong></summary>

To install the latest compatible CUDA distribution, set the `TORCH_CUDA_VERSION` environment
variable before the `tch-rs` dependency is retrieved with `cargo`.

```shell
export TORCH_CUDA_VERSION=cu121
```

On Windows:

```powershell
$Env:TORCH_CUDA_VERSION = "cu121"
```

For example, running the validation sample for the first time could be done with the following
commands:

```shell
export TORCH_CUDA_VERSION=cu121
cargo run --bin cuda --release
```

**Important:** make sure your driver version is compatible with the selected CUDA version. A CUDA
Toolkit installation is not required since LibTorch ships with the appropriate CUDA runtimes. Having
the latest driver version is recommended, but you can always take a look at the
[toolkit driver version table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id4)
or
[minimum required driver version](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility)
(limited feature-set, might not work with all operations).

</details><br>

Once your installation is complete, you should be able to build/run your project. You can also
validate your installation by running the appropriate `cpu`, `cuda` or `mps` sample as below.

```shell
cargo run --bin cpu --release
cargo run --bin cuda --release
cargo run --bin mps --release
```

_Note: no MPS distribution is available for automatic download at this time, please check out the
[manual instructions](#metal-mps)._

### Manual Download

To install `tch-rs` with a different LibTorch distribution, you will have to manually download the
desired LibTorch distribution. The instructions are detailed in the sections below for each
platform.

| Compute Platform          |              CPU               | GPU | Linux | MacOS | Windows | Android | iOS | WASM |
| :------------------------ | :----------------------------: | :-: | :---: | :---: | :-----: | :-----: | :-: | :--: |
| [CPU](#cpu)               |              Yes               | No  |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
| [CUDA](#cuda)             | Yes <sup>[[1]](#cpu-sup)</sup> | Yes |  Yes  |  No   |   Yes   |   No    | No  |  No  |
| [Metal (MPS)](#metal-mps) |               No               | Yes |  No   |  Yes  |   No    |   No    | No  |  No  |
| Vulkan                    |              Yes               | Yes |  Yes  |  Yes  |   Yes   |   Yes   | No  |  No  |

<sup><a id="cpu-sup">[1]</a> The LibTorch CUDA distribution also comes with CPU support.</sup>

#### CPU

<details open>
<summary><strong>🐧 Linux</strong></summary>

First, download the LibTorch CPU distribution.

```shell
wget -O libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcpu.zip
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
wget -O libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.2.0.zip
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
wget https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.2.0%2Bcpu.zip -OutFile libtorch.zip
Expand-Archive libtorch.zip
```

Then, set the `LIBTORCH` environment variable and append the library to your path as with the
PowerShell commands below before building `burn-tch` or a crate which depends on it.

```powershell
$Env:LIBTORCH = "/absolute/path/to/libtorch/"
$Env:Path += ";/absolute/path/to/libtorch/"
```

</details><br>

#### CUDA

LibTorch 2.2.0 currently includes binary distributions with CUDA 11.8 or 12.1 runtimes. The manual
installation instructions are detailed below.

**CUDA 11.8**

<details open>
<summary><strong>🐧 Linux</strong></summary>

First, download the LibTorch CUDA 11.8 distribution.

```shell
wget -O libtorch.zip https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcu118.zip
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
wget https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.2.0%2Bcu118.zip -OutFile libtorch.zip
Expand-Archive libtorch.zip
```

Then, set the `LIBTORCH` environment variable and append the library to your path as with the
PowerShell commands below before building `burn-tch` or a crate which depends on it.

```powershell
$Env:LIBTORCH = "/absolute/path/to/libtorch/"
$Env:Path += ";/absolute/path/to/libtorch/"
```

</details><br>

**CUDA 12.1**

<details open>
<summary><strong>🐧 Linux</strong></summary>

First, download the LibTorch CUDA 12.1 distribution.

```shell
wget -O libtorch.zip https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcu121.zip
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
wget https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.2.0%2Bcu121.zip -OutFile libtorch.zip
Expand-Archive libtorch.zip
```

Then, set the `LIBTORCH` environment variable and append the library to your path as with the
PowerShell commands below before building `burn-tch` or a crate which depends on it.

```powershell
$Env:LIBTORCH = "/absolute/path/to/libtorch/"
$Env:Path += ";/absolute/path/to/libtorch/"
```

</details><br>

#### Metal (MPS)

There is no official LibTorch distribution with MPS support at this time, so the easiest alternative
is to use a PyTorch installation. This requires a Python installation.

_Note: MPS acceleration is available on MacOS 12.3+._

```shell
pip install torch==2.2.0 numpy==1.26.4 setuptools
export LIBTORCH_USE_PYTORCH=1
export DYLD_LIBRARY_PATH=/path/to/pytorch/lib:$DYLD_LIBRARY_PATH
```

**Note:** if `venv` is used, it should be activated during coding and building,
or the compiler may not work properly.

## Example Usage

For a simple example, check out any of the test programs in [`src/bin/`](./src/bin/). Each program
sets the device to use and performs a simple element-wise addition.

For a more complete example using the `tch` backend, take a loot at the
[Burn mnist example](https://github.com/tracel-ai/burn/tree/main/examples/mnist).

## Too many environment variables?

Try `.cargo/config.toml` ([cargo book](https://doc.rust-lang.org/cargo/reference/config.html#env)).

Instead of setting the environments in your shell, you can manually add them to your `.cargo/config.toml`:

```toml
[env]
LD_LIBRARY_PATH = "/absolute/path/to/libtorch/lib"
LIBTORCH = "/absolute/path/to/libtorch/libtorch"
```

Or use bash commands below:

```bash
mkdir .cargo
cat <<EOF > .cargo/config.toml
[env]
LD_LIBRARY_PATH = "/absolute/path/to/libtorch/lib:$LD_LIBRARY_PATH"
LIBTORCH = "/absolute/path/to/libtorch/libtorch"
EOF
```
This will automatically include the old `LD_LIBRARY_PATH` value in the new one.
