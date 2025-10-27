# Text Generation

> **Note**  
> This example makes use of the HuggingFace [`datasets`](https://huggingface.co/docs/datasets/index)
> library to download the datasets. Make sure you have [Python](https://www.python.org/downloads/)
> installed on your computer.

The example can be run like so:

## CUDA users

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn

# Use the --release flag to really speed up training.
export TORCH_CUDA_VERSION=cu128
cargo run --example text-generation --release
```

## Mac users

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn

# Use the --release flag to really speed up training.
cargo run --example text-generation --release
```
