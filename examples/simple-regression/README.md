# Regression

The example shows you how to:

- Define a custom dataset for regression problems. We implement the
  [California Housing Dataset](https://huggingface.co/datasets/gvlassis/california_housing) from
  HuggingFace hub. The dataset is also available as part of toy regression datasets in
  sklearn[datasets](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).
- Create a data pipeline from a raw dataset to a batched fast DataLoader with min-max feature
  scaling.
- Define a Simple NN model for regression using Burn Modules.

> **Note**  
> This example makes use of the HuggingFace [`datasets`](https://huggingface.co/docs/datasets/index)
> library to download the datasets. Make sure you have [Python](https://www.python.org/downloads/)
> installed on your computer.

The example can be run like so:

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn
# Use the --release flag to really speed up training.
echo "Using flex backend"
cargo run --example regression --release --features flex                   # CPU Flex Backend - f32
echo "Using tch backend"
export TORCH_CUDA_VERSION=cu128                                            # Set the cuda version
cargo run --example regression --release --features tch-gpu                # GPU Tch Backend - f32
cargo run --example regression --release --features tch-cpu                # CPU Tch Backend - f32
echo "Using wgpu backend"
cargo run --example regression --release --features wgpu
```
