# Burn Neural Networks

> Neural network layers, activations and losses for [Burn](https://github.com/tracel-ai/burn)

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-nn.svg)](https://crates.io/crates/burn-nn)
[![Documentation](https://docs.rs/burn-nn/badge.svg)](https://docs.rs/burn-nn)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

Every layer is a module built from a config, and applications use them through `burn::nn`:

```rust,ignore
use burn::nn::{Linear, LinearConfig};

let linear: Linear = LinearConfig::new(784, 128).init(&device);
let output = linear.forward(input);
```

- Layers: linear, convolution and transposed convolution (1D to 3D), pooling, normalization (batch,
  layer, group, instance, RMS), embeddings, dropout, recurrent layers (LSTM, GRU), attention and
  transformers, positional and rotary encodings, interpolation, and more.
- `activation`: activation functions as modules.
- `loss`: loss functions, from mean squared error and cross-entropy to CTC.
- `Initializer`: weight initialization schemes.

See the [module chapter](https://burn.dev/books/burn/building-blocks/module.html) of the Burn Book.

## Feature Flags

- `std` (default): standard library support. Without it the crate is `no_std` with `alloc`.
- `tracing`: instrument operations with the `tracing` crate.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
