# Text Classification

This project provides an example implementation for training and inferencing text classification
models on AG News and DbPedia datasets using the Rust-based Burn Deep Learning Library.
It also provides an example of finetuning using [LoRA](https://arxiv.org/pdf/2106.09685) (See [this section](#finetuning-using-lora)).

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

## Flex backend

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn

# Use the --release flag to really speed up training.

# AG News
cargo run -p text-classification --example ag-news-train --release --features flex   # Train on the ag news dataset
cargo run -p text-classification --example ag-news-infer --release --features flex   # Run inference on the ag news dataset

# DbPedia
cargo run -p text-classification --example db-pedia-train --release --features flex  # Train on the db pedia dataset
cargo run -p text-classification --example db-pedia-infer --release --features flex  # Run inference db pedia dataset
```

## Profiling

The `profile` example measures where the AG News classifier's inference spends its device time,
component by component, and prints the report. It needs a CubeCL backend:

```bash
cargo run -p text-classification --example profile --release --features cuda
# Also save the measurement, to hold a later run against with `burn-instrument-report --against`.
cargo run -p text-classification --example profile --release --features cuda -- now.json
# In half precision, with `f16`, `bf16` or `flex32`.
cargo run -p text-classification --example profile --release --features cuda,f16
```

See [Profiling](../../burn-book/src/performance/profiling.md) in the Burn Book for how to read it.

## WGPU backend

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn

# Use the --release flag to really speed up training.

# AG News
cargo run -p text-classification --example ag-news-train --release --features wgpu   # Train on the ag news dataset
cargo run -p text-classification --example ag-news-infer --release --features wgpu   # Run inference on the ag news dataset

# DbPedia
cargo run -p text-classification --example db-pedia-train --release --features wgpu  # Train on the db pedia dataset
cargo run -p text-classification --example db-pedia-infer --release --features wgpu  # Run inference db pedia dataset
```

## CUDA backend

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn

# Use the --release flag to really speed up training.
# Add the f16 feature to run in f16. 

# AG News
cargo run -p text-classification --example ag-news-train --release --features cuda   # Train on the ag news dataset
cargo run -p text-classification --example ag-news-infer --release --features cuda   # Run inference on the ag news dataset
```

## Metal backend

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn

# Use the --release flag to really speed up training.
# Add the f16 feature to run in f16. 

# AG News
cargo run -p text-classification --example ag-news-train --release --features metal   # Train on the ag news dataset
cargo run -p text-classification --example ag-news-infer --release --features metal   # Run inference on the ag news dataset
```

# Finetuning Using LoRA

This example finetunes a pre-trained text classification model on the AG News dataset using [LoRA](https://arxiv.org/pdf/2106.09685).
To run it, you need to provide a `model.bpk`file containing the weights of the pre-trained text classification model.
Such a file can be obtained by running the `db-pedia-train` example using one of the commands above.
You can then run the finetuning example using any of the backends listed above. For example, with Wgpu:

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn

# Use the --release flag to really speed up training.

# AG News
cargo run -p text-classification --example ag-news-finetune --release --features wgpu   # Finetune pre-trained `model.bpk` weights on the ag news dataset
```