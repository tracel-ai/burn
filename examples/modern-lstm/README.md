# Advanced LSTM Implementation with Burn
A sophisticated implementation of Long Short-Term Memory (LSTM) networks in Burn, featuring state-of-the-art architectural enhancements and optimizations. This implementation includes bidirectional processing capabilities and advanced regularization techniques. More details can be found at the [PyTorch implementation](https://github.com/shiv08/Advanced-LSTM-Implementation-with-PyTorch).

`LstmNetwork` is the top-level module with bidirectional support and output projection. It can support multiple LSTM variants by setting appropriate `bidirectional` and `num_layers`：
* LSTM: `num_layers = 1` and `bidirectional = false`
* Stacked LSTM: `num_layers > 1` and `bidirectional = false`
* Bidirectional LSTM: `num_layers = 1` and `bidirectional = true`
* Bidirectional Stacked LSTM: `num_layers > 1` and `bidirectional = true`

This implementation is complementary to Burn's official LSTM, users can choose either one depends on the project's specific needs.

## Usage


## Training

```sh
# Cuda backend
cargo run --example train --release --features cuda-jit

# Wgpu backend
cargo run --example train --release --features wgpu

# Tch GPU backend
export TORCH_CUDA_VERSION=cu121 # Set the cuda version
cargo run --example train --release --features tch-gpu

# Tch CPU backend
cargo run --example train --release --features tch-cpu

# NdArray backend (CPU)
cargo run --example train --release --features ndarray
cargo run --example train --release --features ndarray-blas-openblas
cargo run --example train --release --features ndarray-blas-netlib
```


### Inference

```sh
cargo run --example infer --release --features cuda-jit
```
