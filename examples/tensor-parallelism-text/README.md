# Tensor Parallelism on Text Classification

Runs the transformer the `text-classification` example trains on AG News with every encoder
layer split over several devices, Megatron style. The devices form one sharded compute device,
and every layer's weights are split across it:

- Attention: the query, key and value projections by output column, so each member computes whole
  heads, and the output projection by input row.
- Feed-forward: the first projection by column and the second by row, so the hidden activation
  never leaves its member.

Each row-split projection ends in one all-reduce, two per layer. The embeddings, layer norms and
classification head stay replicated on every member. `src/split.rs` is the whole plan. Each split
weight is read from the checkpoint and cut on the host, and each member receives only its chunk.

Train the single-device model first. It saves to `/tmp/text-classification-ag-news`, and the
example takes another directory as its argument. Without one it uses fresh weights, which classify
at random but split the same way.

```bash
cargo run -p text-classification --example ag-news-train --release --features cuda
```

Then:

```bash
# Every CUDA GPU, checked against the same weights on the first one alone.
cargo run -p tensor-parallelism-text --example tensor-parallelism-infer --release --features cuda
```

`--features rocm` or `--features vulkan` uses those GPUs instead. Without a GPU feature, the model
splits over two Flex devices. The number of devices must divide the model's 4 heads.
