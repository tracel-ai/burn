# Regression

The example shows you how to:

- Define a custom dataset for regression problems. We implement the  Diabetes Toy Dataset from HuggingFace hub. (Add links)
- Create a data pipeline from a raw dataset to a batched fast DataLoader.
- Define a Simple Linear Regression model using Burn Modules.

The example can be run like so:

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn
# Use the --release flag to really speed up training.
echo "Using ndarray backend"
cargo run --example regression --release --features ndarray                # CPU NdArray Backend - f32 - single thread
cargo run --example regression --release --features ndarray-blas-openblas  # CPU NdArray Backend - f32 - blas with openblas
cargo run --example regression --release --features ndarray-blas-netlib    # CPU NdArray Backend - f32 - blas with netlib
echo "Using tch backend"
export TORCH_CUDA_VERSION=cu113                                       # Set the cuda version
cargo run --example regression --release --features tch-gpu                # GPU Tch Backend - f32
cargo run --example regression --release --features tch-cpu                # CPU Tch Backend - f32
echo "Using wgpu backend"
cargo run --example regression --release --features wgpu
```
