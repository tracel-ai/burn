# Lstm Timeseries Forecasting 

The example shows you how to:

- Define a custom dataset for timeseries prediction. We implement the [timeseries-1d-stocks Dataset](https://huggingface.co/datasets/edarchimbaud/timeseries-1d-stocks) 
from HuggingFace hub.
- Create a data pipeline from a raw dataset to a batched fast DataLoader with time step windows and min-max feature scaling.
- Define a LSTM network for time series prediction using Burn Modules.

> **Note**  
> This example makes use of the HuggingFace [`datasets`](https://huggingface.co/docs/datasets/index)
> library to download the datasets. Make sure you have [Python](https://www.python.org/downloads/)
> installed on your computer.

The example can be run like so:

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn
# Use the --release flag to really speed up training.
echo "Using ndarray backend"
cargo run --example lstm --release --features ndarray                # CPU NdArray Backend - f32 - single thread
cargo run --example lstm --release --features ndarray-blas-openblas  # CPU NdArray Backend - f32 - blas with openblas
cargo run --example lstm --release --features ndarray-blas-netlib    # CPU NdArray Backend - f32 - blas with netlib
echo "Using tch backend"
export TORCH_CUDA_VERSION=cu121                                            # Set the cuda version
cargo run --example lstm --release --features tch-gpu                # GPU Tch Backend - f32
cargo run --example lstm --release --features tch-cpu                # CPU Tch Backend - f32
echo "Using wgpu backend"
cargo run --example lstm --release --features wgpu
```
