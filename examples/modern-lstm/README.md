# Advanced LSTM Implementation with Burn

A more advanced implementation of Long Short-Term Memory (LSTM) networks in Burn with combined
weight matrices for the input and hidden states, based on the
[PyTorch implementation](https://github.com/shiv08/Advanced-LSTM-Implementation-with-PyTorch).

`LstmNetwork` is the top-level module with bidirectional and regularization support. The LSTM
variants differ by `bidirectional` and `num_layers` settings：

- LSTM: `num_layers = 1` and `bidirectional = false`
- Stacked LSTM: `num_layers > 1` and `bidirectional = false`
- Bidirectional LSTM: `num_layers = 1` and `bidirectional = true`
- Bidirectional Stacked LSTM: `num_layers > 1` and `bidirectional = true`

This implementation is complementary to Burn's official LSTM, users can choose either one depends on
the project's specific needs.

## Usage

## Training

```sh
# Cuda backend
cargo run --example lstm-train --release --features cuda

# Wgpu backend
cargo run --example lstm-train --release --features wgpu

# Tch GPU backend
export TORCH_CUDA_VERSION=cu121 # Set the cuda version
cargo run --example lstm-train --release --features tch-gpu

# Tch CPU backend
cargo run --example lstm-train --release --features tch-cpu

# NdArray backend (CPU)
cargo run --example lstm-train --release --features ndarray
cargo run --example lstm-train --release --features ndarray-blas-openblas
cargo run --example lstm-train --release --features ndarray-blas-netlib
```

### Inference

```sh
cargo run --example lstm-infer --release --features cuda
```
