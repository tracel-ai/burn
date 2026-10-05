# Burn Train

> Training and evaluation for [Burn](https://github.com/tracel-ai/burn) models

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-train.svg)](https://crates.io/crates/burn-train)
[![Documentation](https://docs.rs/burn-train/badge.svg)](https://docs.rs/burn-train)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

Applications enable the `train` feature of `burn` and use this crate as `burn::train`:

```toml
burn = { version = "0.22", features = ["train", "wgpu"] }
```

- `Learner` bundles a model, optimizer and learning rate scheduler, and `SupervisedTraining` runs it
  over training and validation data loaders on one or several devices.
- `metric`: loss, accuracy, precision and recall, F-scores, AUROC, BLEU, CER, WER, perplexity, system
  usage and more; with the `vision` feature, image metrics such as PSNR, SSIM, LPIPS and FID.
- `renderer`: a terminal dashboard (`tui` feature) or plain CLI output for training progress.
- `checkpoint`: periodic and metric-based checkpointing, and resuming from a checkpoint.
- Early stopping, interruption and an `Evaluator` for test sets.

Failures on a device during training are returned as errors rather than panics.

See the [learner](https://burn.dev/books/burn/building-blocks/learner.html) and
[metric](https://burn.dev/books/burn/building-blocks/metric.html) chapters of the Burn Book, and the
[guide](https://burn.dev/books/burn/basic-workflow/training.html) for a complete example.

## Feature Flags

- `tui` (default): terminal dashboard.
- `sys-metrics` (default): CPU, memory and GPU usage metrics.
- `vision`: image quality metrics.
- `rl`: reinforcement learning training through [burn-rl](https://github.com/tracel-ai/burn/tree/main/crates/burn-rl).
- `tracing`: instrument operations with the `tracing` crate.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
