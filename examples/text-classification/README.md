# Text Classification

This project provides an example implementation for training and inferencing text classification
models on AG News and DbPedia datasets using the Rust-based Burn Deep Learning Library.

## Dataset Details

- AG News: The AG News dataset is a collection of news articles from more than 2000 news sources.
  This library helps you load and process this dataset, categorizing articles into four classes:
  "World", "Sports", "Business", and "Technology".

- DbPedia: The DbPedia dataset is a large multi-class text classification dataset extracted from
  Wikipedia. This library helps you load and process this dataset, categorizing articles into 14
  classes including "Company", "Educational Institution", "Artist", among others.

## Usage

## CUDA users

NOTE: enable f16 feature if your CUDA device supports FP16 (half precision) operations.

```bash
git clone https://github.com/burn-rs/burn.git
cd burn
# Use the --release flag to really speed up training.
export TORCH_CUDA_VERSION=cu117  # Set the cuda version
cargo run --example ag-news-train --release --features f16   # Train on the ag news dataset
cargo run --example ag-news-infer --release # Run inference on the ag news dataset

cargo run --example db-pedia-train --release --features f16  # Train on the db pedia dataset
cargo run --example db-pedia-infer --release   # Run inference db pedia dataset
```

## Mac users

NOTE: Enabling f16 feature can cause the program to generate NaN on Mac.

```bash
git clone https://github.com/burn-rs/burn.git
cd burn
# Use the --release flag to really speed up training.
export TORCH_CUDA_VERSION=cu117  # Set the cuda version
cargo run --example ag-news-train --release   # Train on the ag news dataset
cargo run --example ag-news-infer --release # Run inference on the ag news dataset

cargo run --example db-pedia-train --release  # Train on the db pedia dataset
cargo run --example db-pedia-infer --release   # Run inference db pedia dataset
```
