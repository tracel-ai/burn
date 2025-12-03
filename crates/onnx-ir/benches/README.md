# ONNX-IR Benchmarks

Benchmarks comparing memory-mapped (mmap) vs standard ONNX model loading.

## Setup

Before running benchmarks, download the test models from Hugging Face:

```bash
cd crates/onnx-ir
uv run benches/generate_bench_model.py
```

This downloads:

- **MiniLM** (~86 MB) - sentence transformer model
- **CLIP Vision** (~336 MB) - vision encoder model

Models are saved to your system's temp directory (`/tmp/onnx_ir_bench_models/` on Unix).

## Running Benchmarks

```bash
# With mmap enabled (default)
cargo bench --bench mmap_loading

# Without mmap
cargo bench --bench mmap_loading --no-default-features
```

## What's Measured

The benchmark compares three loading methods:

| Method         | Description                                  |
| -------------- | -------------------------------------------- |
| `parse_file`   | Load from file path (uses mmap when enabled) |
| `parse_bytes`  | Load from pre-read bytes (no mmap)           |
| `parse_reader` | Load from a reader (no mmap)                 |

Metrics tracked:

- **Time**: How long parsing takes
- **Throughput**: GB/s processing speed
- **Memory**: Peak allocation, total allocations, deallocations

## Expected Results

With mmap enabled, `parse_file` should show:

- **Lower peak memory** - tensor data stays in mmap'd region instead of heap
- **Lower total allocations** - no copy of file contents needed
- **Similar or better speed** - no upfront memory copy

Example results for CLIP Vision (336 MB):

| Metric      | mmap ON  | mmap OFF |
| ----------- | -------- | -------- |
| Peak memory | ~411 MB  | ~763 MB  |
| Total alloc | ~2.16 GB | ~2.51 GB |
| Time        | ~100ms   | ~104ms   |

## Requirements

- [uv](https://github.com/astral-sh/uv) - for running the Python download script
- Python 3.11+ with `onnx` and `huggingface_hub` (installed automatically by uv)
