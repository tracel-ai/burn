<div align="center">
<img src="https://raw.githubusercontent.com/burn-rs/burn/main/assets/logo-burn-full.png" width="200px"/>

[![Discord](https://img.shields.io/discord/1038839012602941528.svg?color=7289da&&logo=discord)](https://discord.gg/uPEBbYYDB6)
[![Current Crates.io Version](https://img.shields.io/crates/v/burn.svg)](https://crates.io/crates/burn)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://burn.dev/docs/burn)
[![Test Status](https://github.com/burn-rs/burn/actions/workflows/test.yml/badge.svg)](https://github.com/burn-rs/burn/actions/workflows/test.yml)
[![CodeCov](https://codecov.io/gh/burn-rs/burn/branch/main/graph/badge.svg)](https://codecov.io/gh/burn-rs/burn)
[![Rust Version](https://img.shields.io/badge/Rust-1.71.0+-blue)](https://releases.rs/docs/1.71.0)
![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)

This library strives to serve as a comprehensive **deep learning framework**, offering exceptional
flexibility and written in Rust. Our objective is to cater to both researchers and practitioners by
simplifying the process of experimenting, training, and deploying models.

<div align="left">

## Features

- Customizable, intuitive and user-friendly neural network [module](https://burn-rs.github.io/book/building-blocks/module.html) 🔥
- Comprehensive [training](https://burn-rs.github.io/book/building-blocks/learner.html) tools, including `metrics`, `logging`, and `checkpointing`
  📈
- Versatile [Tensor](https://burn-rs.github.io/book/building-blocks/tensor.html) crate equipped with pluggable backends 🔧
  - [Torch](https://github.com/burn-rs/burn/tree/main/burn-tch) backend, supporting both CPU and GPU
    🚀
  - [Ndarray](https://github.com/burn-rs/burn/tree/main/burn-ndarray) backend with
    [`no_std`](#support-for-no_std) compatibility, ensuring universal platform adaptability 👌
  - [WebGPU](https://github.com/burn-rs/burn/tree/main/burn-wgpu) backend, offering cross-platform,
    browser-inclusive, GPU-based computations 🌐
  - [Candle](https://github.com/burn-rs/burn/tree/main/burn-candle) backend 🕯️
  - [Autodiff](https://github.com/burn-rs/burn/tree/main/burn-autodiff) backend that enables
    differentiability across all backends 🌟
- [Dataset](https://github.com/burn-rs/burn/tree/main/burn-dataset) crate containing a diverse range
  of utilities and sources 📚
- [Import](https://github.com/burn-rs/burn/tree/main/burn-import) crate that simplifies the
  integration of pretrained models 📦

## Get Started

### The Burn Book 🔥

To begin working effectively with `burn`, it is crucial to understand its key components and philosophy.
For detailed examples and explanations covering every facet of the framework, please refer to [The Burn Book 🔥](https://burn-rs.github.io/book/).

### Pre-trained Models

We keep an updated and curated list of models and examples built with Burn, see the [burn-rs/models](https://github.com/burn-rs/models) repository for more details.

### Examples

Here is a code snippet showing how intuitive the framework is to use, where we declare a position-wise feed-forward module along with its forward pass.

```rust
use burn::nn;
use burn::module::Module;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct PositionWiseFeedForward<B: Backend> {
    linear_inner: Linear<B>,
    linear_outer: Linear<B>,
    dropout: Dropout,
    gelu: GELU,
}

impl<B: Backend> PositionWiseFeedForward<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear_inner.forward(input);
        let x = self.gelu.forward(x);
        let x = self.dropout.forward(x);

        self.linear_outer.forward(x)
    }
}
```

For more practical insights, you can clone the repository and experiment with the following examples:

- [MNIST](https://github.com/burn-rs/burn/tree/main/examples/mnist) train a model on CPU/GPU using
  different backends.
- [MNIST Inference Web](https://github.com/burn-rs/burn/tree/main/examples/mnist-inference-web) run
  trained model in the browser for inference.
- [Text Classification](https://github.com/burn-rs/burn/tree/main/examples/text-classification)
  train a transformer encoder from scratch on GPU.
- [Text Generation](https://github.com/burn-rs/burn/tree/main/examples/text-generation) train an
  autoregressive transformer from scratch on GPU.

## Supported Platforms

### [Burn-ndarray][1] Backend

| Option     | CPU | GPU | Linux | MacOS | Windows | Android | iOS | WASM |
| :--------- | :-: | :-: | :---: | :---: | :-----: | :-----: | :-: | :--: |
| Pure Rust  | Yes | No  |  Yes  |  Yes  |   Yes   |   Yes   | Yes | Yes  |
| Accelerate | Yes | No  |  No   |  Yes  |   No    |   No    | Yes |  No  |
| Netlib     | Yes | No  |  Yes  |  Yes  |   Yes   |   No    | No  |  No  |
| Openblas   | Yes | No  |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |

### [Burn-tch][2] Backend

| Option | CPU | GPU | Linux | MacOS | Windows | Android | iOS | WASM |
| :----- | :-: | :-: | :---: | :---: | :-----: | :-----: | :-: | :--: |
| CPU    | Yes | No  |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
| CUDA   | No  | Yes |  Yes  |  No   |   Yes   |   No    | No  |  No  |
| MPS    | No  | Yes |  No   |  Yes  |   No    |   No    | No  |  No  |
| Vulkan | Yes | Yes |  Yes  |  Yes  |   Yes   |   Yes   | No  |  No  |

### [Burn-wgpu][3] Backend

| Option    | CPU | GPU | Linux | MacOS | Windows | Android | iOS | WASM |
| :-------- | :-: | :-: | :---: | :---: | :-----: | :-----: | :-: | :--: |
| Metal     | No  | Yes |  No   |  Yes  |   No    |   No    | Yes |  No  |
| Vulkan    | Yes | Yes |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
| OpenGL    | No  | Yes |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
| WebGpu    | No  | Yes |  No   |  No   |   No    |   No    | No  | Yes  |
| Dx11/Dx12 | No  | Yes |  No   |  No   |   Yes   |   No    | No  |  No  |

[1]: https://github.com/burn-rs/burn/tree/main/burn-ndarray
[2]: https://github.com/burn-rs/burn/tree/main/burn-tch
[3]: https://github.com/burn-rs/burn/tree/main/burn-wgpu

## Support for `no_std`

Burn, including its `burn-ndarray` backend, can work in a `no_std` environment, provided `alloc` is
available for the inference mode. To accomplish this, simply turn off the default features in `burn`
and `burn-ndarray` (which is the minimum requirement for running the inference mode). You can find a
reference example in
[burn-no-std-tests](https://github.com/burn-rs/burn/tree/main/burn-no-std-tests).

The `burn-core` and `burn-tensor` crates also support `no_std` with `alloc`. These crates can be
directly added as dependencies if necessary, as they are reexported by the `burn` crate.

Please be aware that when using the `no_std` mode, a random seed will be generated at build time if
one hasn't been set using the `Backend::seed` method. Also, the
[spin::mutex::Mutex](https://docs.rs/spin/latest/spin/mutex/struct.Mutex.html) is used instead of
[std::sync::Mutex](https://doc.rust-lang.org/std/sync/struct.Mutex.html) in this mode.

## Contributing

Before contributing, please take a moment to review our
[code of conduct](https://github.com/burn-rs/burn/tree/main/CODE-OF-CONDUCT.md). It's also highly
recommended to read our
[architecture document](https://github.com/burn-rs/burn/tree/main/ARCHITECTURE.md), which explains
our architectural decisions. Please see more details in our [contributing guide](/CONTRIBUTING.md).

## Disclaimer

Burn is currently in active development, and there will be breaking changes. While any resulting
issues are likely to be easy to fix, there are no guarantees at this stage.

## Sponsors

Thanks to all current sponsors 🙏.

<a href="https://github.com/smallstepman"><img src="https://github.com/smallstepman.png" width="60px" style="border-radius: 50%;" alt="smallstepman" /></a>
<a href="https://github.com/premAI-io"><img src="https://github.com/premAI-io.png" width="60px" style="border-radius: 50%;" alt="premAI-io" /></a>

## License

Burn is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
See [LICENSE-APACHE](./LICENSE-APACHE) and [LICENSE-MIT](./LICENSE-MIT) for details. Opening a pull
request is assumed to signal agreement with these licensing terms.
