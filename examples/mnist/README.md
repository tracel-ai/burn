# MNIST

The example is showing you how to:

- Define your own custom module (MLP).
- Create the data pipeline from a raw dataset to a batched multi-threaded fast DataLoader.
- Configure a learner to display and log metrics as well as to keep training checkpoints.

The example can be run like so:

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn
# Use the --release flag to really speed up training.
echo "Using ndarray backend"
cargo run --example mnist --release --features ndarray                # CPU NdArray Backend - f32 - single thread
cargo run --example mnist --release --features ndarray-blas-openblas  # CPU NdArray Backend - f32 - blas with openblas
cargo run --example mnist --release --features ndarray-blas-netlib    # CPU NdArray Backend - f32 - blas with netlib
echo "Using tch backend"
export TORCH_CUDA_VERSION=cu128                                       # Set the cuda version
cargo run --example mnist --release --features tch-gpu                # GPU Tch Backend - f32
cargo run --example mnist --release --features tch-cpu                # CPU Tch Backend - f32
echo "Using vulkan backend"
cargo run --example mnist --release --features vulkan
```
