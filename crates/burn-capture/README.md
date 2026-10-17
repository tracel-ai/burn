# Burn Capture

> A [Burn](https://github.com/tracel-ai/burn) backend that records operation graphs instead of executing them

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-capture.svg)](https://crates.io/crates/burn-capture)
[![Documentation](https://docs.rs/burn-capture/badge.svg)](https://docs.rs/burn-capture)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

Tensors on a capture device are recorded rather than computed. Each capture scope runs ordinary
tensor code, declares the graph's inputs and outputs, and returns a `CapturedGraph`: the recorded
[burn-ir](https://github.com/tracel-ai/burn/tree/main/crates/burn-ir) `GraphIr` plus the recorded
tensor data, such as model weights. This is for tools that need a model's computation as a graph,
such as exporters, rather than its results.

## Usage

Enable Burn's `capture` feature, plus `extension` to read the IDs of the tensors that bound the
graph:

```toml
burn = { version = "0.22", features = ["capture", "extension", "flex"] }
burn-capture = "0.22"
```

```rust,ignore
use burn::nn::LinearConfig;
use burn::prelude::*;
use burn_capture::{CaptureBackend, TensorId};

fn id<const D: usize>(tensor: &Tensor<D>) -> TensorId {
    tensor.clone().try_into_primitive::<CaptureBackend>().unwrap().id()
}

let model = LinearConfig::new(4, 2).init(&Device::flex());
let device = Device::capture();

let captured = device.capture_scope(|scope| {
    let model = model.clone().to_device(&device);
    // A runtime input comes from outside the scope; tensors created inside it are recorded as
    // constants.
    let input = Tensor::<2>::zeros([1, 4], &Device::flex()).to_device(&device);
    let output = model.forward(input.clone());
    scope.complete([id(&input)], [id(&output)])
})?;

// `captured.graph` holds the operations and their inputs and outputs; `captured.values` holds
// the recorded tensor data, such as the model's weights.
```

Module parameters are not graph inputs: their values are recorded. A capture device can be reused
for any number of sequential scopes, one at a time.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
