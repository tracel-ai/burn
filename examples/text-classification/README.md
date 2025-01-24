# Text Classification

This project provides an example implementation for training and inferencing text classification
models on AG News and DbPedia datasets using the Rust-based Burn Deep Learning Library.

> **Note**  
> This example makes use of the HuggingFace [`datasets`](https://huggingface.co/docs/datasets/index)
> library to download the datasets. Make sure you have [Python](https://www.python.org/downloads/)
> installed on your computer.

## Dataset Details

- AG News: The AG News dataset is a collection of news articles from more than 2000 news sources.
  This library helps you load and process this dataset, categorizing articles into four classes:
  "World", "Sports", "Business", and "Technology".

- DbPedia: The DbPedia dataset is a large multi-class text classification dataset extracted from
  Wikipedia. This library helps you load and process this dataset, categorizing articles into 14
  classes including "Company", "Educational Institution", "Artist", among others.

# Usage

## Torch GPU backend

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn

# Use the --release flag to really speed up training.
# Use the f16 feature if your CUDA device supports FP16 (half precision) operations. May not work well on every device.

export TORCH_CUDA_VERSION=cu121  # Set the cuda version (CUDA users)

# AG News
cargo run --example ag-news-train --release --features tch-gpu   # Train on the ag news dataset
cargo run --example ag-news-infer --release --features tch-gpu   # Run inference on the ag news dataset

# DbPedia
cargo run --example db-pedia-train --release --features tch-gpu  # Train on the db pedia dataset
cargo run --example db-pedia-infer --release --features tch-gpu  # Run inference db pedia dataset
```

## Torch CPU backend

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn

# Use the --release flag to really speed up training.

# AG News
cargo run --example ag-news-train --release --features tch-cpu   # Train on the ag news dataset
cargo run --example ag-news-infer --release --features tch-cpu   # Run inference on the ag news dataset

# DbPedia
cargo run --example db-pedia-train --release --features tch-cpu  # Train on the db pedia dataset
cargo run --example db-pedia-infer --release --features tch-cpu  # Run inference db pedia dataset
```

## ndarray backend

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn

# Use the --release flag to really speed up training.

# Replace ndarray by ndarray-blas-netlib, ndarray-blas-openblas or ndarray-blas-accelerate for different matmul techniques

# AG News
cargo run --example ag-news-train --release --features ndarray   # Train on the ag news dataset
cargo run --example ag-news-infer --release --features ndarray   # Run inference on the ag news dataset

# DbPedia
cargo run --example db-pedia-train --release --features ndarray  # Train on the db pedia dataset
cargo run --example db-pedia-infer --release --features ndarray  # Run inference db pedia dataset
```

## WGPU backend

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn

# Use the --release flag to really speed up training.

# AG News
cargo run --example ag-news-train --release --features wgpu   # Train on the ag news dataset
cargo run --example ag-news-infer --release --features wgpu   # Run inference on the ag news dataset

# DbPedia
cargo run --example db-pedia-train --release --features wgpu  # Train on the db pedia dataset
cargo run --example db-pedia-infer --release --features wgpu  # Run inference db pedia dataset
```

## CUDA backend

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn

# Use the --release flag to really speed up training.

# AG News
cargo run --example ag-news-train --release --features cuda   # Train on the ag news dataset
cargo run --example ag-news-infer --release --features cuda   # Run inference on the ag news dataset
```
