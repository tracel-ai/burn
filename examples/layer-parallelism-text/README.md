# Layer Parallelism on Text Classification

Splits the transformer the `text-classification` example trains on AG News by whole layers, one
device per GPU: the embeddings on the first, the encoder layers shared out, the head on the last.
Each layer is built on its device and its weights load straight there, so the model never exists
whole on one device.

Train the single-device model first. It saves to `/tmp/text-classification-ag-news`, and both
examples below take another directory as their argument:

```bash
cargo run -p text-classification --example ag-news-train --release --features cuda
```

Then:

```bash
# Split the trained model and check it against the single-device one on the AG News test set.
cargo run -p layer-parallelism-text --example layer-parallelism-infer --release --features cuda,vulkan

# Fine-tune the split for DbPedia, with a new head, and report test accuracy before and after.
cargo run -p layer-parallelism-text --example layer-parallelism-finetune --release --features cuda,vulkan
```

`--features cuda,vulkan` uses every GPU, NVIDIA ones through CUDA. Without a GPU feature, inference
splits across Flex and the CPU backend, and fine-tuning runs on Flex alone. The datasets download
through Hugging Face, which needs Python.

Layers run one after another, so this buys capacity, not speed.
