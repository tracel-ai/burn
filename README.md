<div align="center">
<img src="https://raw.githubusercontent.com/burn-rs/burn/main/assets/logo-burn-full.png" width="200px"/>

[![Discord](https://img.shields.io/discord/1038839012602941528.svg?color=7289da&&logo=discord)](https://discord.gg/uPEBbYYDB6)
[![Test Status](https://github.com/burn-rs/burn/actions/workflows/test.yml/badge.svg)](https://github.com/burn-rs/burn/actions/workflows/test.yml)
[![Documentation](https://docs.rs/burn/badge.svg)](https://docs.rs/burn)
[![Current Crates.io Version](https://img.shields.io/crates/v/burn.svg)](https://crates.io/crates/burn)
[![Rust Version](https://img.shields.io/badge/Rust-1.65.0+-blue)](https://releases.rs/docs/1.65.0)
![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)

This library strives to serve as a comprehensive **deep learning framework**, offering exceptional
flexibility and written in Rust. Our objective is to cater to both researchers and practitioners by
simplifying the process of experimenting, training, and deploying models.

<div align="left">

## Features

- Customizable, user-friendly neural network [module](#module) 🔥
- Comprehensive [training](#learner) tools, inclusive of `metrics`, `logging`, and `checkpointing`
  📈
- Versatile [Tensor](#tensor) crate equipped with pluggable backends 🔧
  - [Torch](https://github.com/burn-rs/burn/tree/main/burn-tch) backend, supporting both CPU and GPU
    🚀
  - [Ndarray](https://github.com/burn-rs/burn/tree/main/burn-ndarray) backend with
    [`no_std`](#support-for-no_std) compatibility, ensuring universal platform adaptability 👌
  - [WebGPU](https://github.com/burn-rs/burn/tree/main/burn-wgpu) backend, offering cross-platform,
    browser-inclusive, GPU-based computations 🌐
  - [Autodiff](https://github.com/burn-rs/burn/tree/main/burn-autodiff) backend that enables
    differentiability across all backends 🌟
- [Dataset](https://github.com/burn-rs/burn/tree/main/burn-dataset) crate containing a diverse range
  of utilities and sources 📚
- [Import](https://github.com/burn-rs/burn/tree/main/burn-import) crate that simplifies the
  integration of pretrained models 📦

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

## Pre-trained Models

We keep an updated and curated list of models and examples built with Burn, see the [burn-rs/models](https://github.com/burn-rs/models) repository for more details.

## Get Started

The best way to get started with `burn` is to clone the repo and play with the
[examples](#examples). This may also be a good idea to take a look the main
[components](#components) of `burn` to get a quick overview of the fundamental building blocks. If
you're interested in how the framework works, you can read our
[architecture document](https://github.com/burn-rs/burn/tree/main/ARCHITECTURE.md).

### Examples

- [MNIST](https://github.com/burn-rs/burn/tree/main/examples/mnist) train a model on CPU/GPU using
  different backends.
- [MNIST Inference Web](https://github.com/burn-rs/burn/tree/main/examples/mnist-inference-web) run
  trained model in the browser for inference.
- [Text Classification](https://github.com/burn-rs/burn/tree/main/examples/text-classification)
  train a transformer encoder from scratch on GPU.
- [Text Generation](https://github.com/burn-rs/burn/tree/main/examples/text-generation) train an
  autoregressive transformer from scratch on GPU.

### Components

Understanding the key components and philosophy of `burn` can greatly help when beginning to work
with the framework.

#### Backend

Nearly everything in `burn` is based on the `Backend` trait, which enables you to run tensor
operations using different implementations without having to modify your code. While a backend may
not necessarily have autodiff capabilities, the `ADBackend` trait specifies when autodiff is needed.
This trait not only abstracts operations but also tensor, device and element types, providing each
backend the flexibility they need. It's worth noting that the trait assumes eager mode since `burn`
fully supports dynamic graphs. However, we may create another API to assist with integrating
graph-based backends, without requiring any changes to the user's code.

#### Tensor

At the core of burn lies the `Tensor` struct, which encompasses multiple types of tensors, including
`Float`, `Int`, and `Bool`. The element types of these tensors are specified by the backend and are
usually designated as a generic argument (e.g., `NdArrayBackend<f32>`). Although the same struct is
used for all tensors, the available methods differ depending on the tensor kind. You can specify the
desired tensor kind by setting the third generic argument, which defaults to `Float`. The first
generic argument specifies the backend, while the second specifies the number of dimensions.

```rust
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, Int};

fn function<B: Backend>(tensor_float: Tensor<B, 2>) {
    let _tensor_bool = tensor_float.clone().equal_elem(2.0); // Tensor<B, 2, Bool>
    let _tensor_int = tensor_float.argmax(1); // Tensor<B, 2, Int>
}
```

As demonstrated in the previous example, nearly all operations require owned tensors as parameters,
which means that calling `Clone` explicitly is necessary when reusing the same tensor multiple
times. However, there's no need to worry since the tensor's data won't be copied, it will be flagged
as readonly when multiple tensors use the same allocated memory. This enables backends to reuse
tensor data when possible, similar to a copy-on-write pattern, while remaining completely
transparent to the user.

#### Autodiff

The 'Backend' trait is highly flexible, enabling backpropagation to be implemented using a simple
backend decorator, which makes any backend differentiable.

```rust
use burn::tensor::backend::{ADBackend, Backend};
use burn::tensor::{Distribution, Tensor};
use burn_autodiff::ADBackendDecorator;
use burn_ndarray::NdArrayBackend;

fn linear<B: Backend>(x: Tensor<B, 2>, weight: Tensor<B, 2>, bias: Tensor<B, 2>) -> Tensor<B, 2> {
    x.matmul(weight) + bias
}

fn main() {
    type Backend = NdArrayBackend<f32>;

    let weight = Tensor::random([3, 3], Distribution::Default);
    let bias = Tensor::zeros([1, 3]);
    let x = Tensor::random([3, 3], Distribution::Default);

    let y = linear::<Backend>(x.clone(), weight.clone(), bias.clone());
    // y.backward() // Method backward doesn't exist

    let y = linear::<ADBackendDecorator<Backend>>(
        Tensor::from_inner(x),
        Tensor::from_inner(weight).require_grad(),
        Tensor::from_inner(bias).require_grad(),
    );
    let grads = y.backward(); // Method exists
}

```

#### Module

The `Module` derive allows you to create your own neural network modules, similar to PyTorch. The
derive function only generates the necessary methods to essentially act as a parameter container for
your type, it makes no assumptions about how the forward pass is declared.

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

Note that all fields declared in the struct must also implement the `Module` trait. The `Tensor`
struct doesn't implement `Module`, but `Param<Tensor<B, D>>` does.

#### Config

The `Config` derive lets you define serializable and deserializable configurations or
hyper-parameters for your [modules](#module) or any components.

```rust
use burn::config::Config;

#[derive(Config)]
pub struct PositionWiseFeedForwardConfig {
    pub d_model: usize,
    pub d_ff: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}
```

The derive also adds useful methods to your config, similar to a builder pattern.

```rust
fn main() {
    let config = PositionWiseFeedForwardConfig::new(512, 2048);
    println!("{}", config.d_model); // 512
    println!("{}", config.d_ff); // 2048
    println!("{}", config.dropout); // 0.1
    let config =  config.with_dropout(0.2);
    println!("{}", config.dropout); // 0.2
}
```

#### Learner

The `Learner` is the main `struct` that let you train a neural network with support for `logging`,
`metric`, `checkpointing` and more. In order to create a learner, you must use the `LearnerBuilder`.

```rust
use burn::train::LearnerBuilder;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::record::DefaultRecordSettings;

fn main() {
    let dataloader_train = ...;
    let dataloader_valid = ...;

    let model = ...;
    let optim = ...;

    let learner = LearnerBuilder::new("/tmp/artifact_dir")
        .metric_train_plot(AccuracyMetric::new())
        .metric_valid_plot(AccuracyMetric::new())
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .with_file_checkpointer::<DefaultRecordSettings>(2)
        .num_epochs(10)
        .build(model, optim);

    let _model_trained = learner.fit(dataloader_train, dataloader_valid);
}
```

See this [example](https://github.com/burn-rs/burn/tree/main/examples/mnist) for a real usage.

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

## Continuous Integration

### Run checks

On Unix systems, run `run-checks.sh` using this command

```
run-checks.sh environment
```

On Windows systems, run `run-checks.ps1` using this command:

```
run-checks.ps1 environment
```

The `environment` argument can assume **ONLY** the following values:

- `std` to perform checks using `libstd`
- `no_std` to perform checks on an embedded environment using `libcore`

If no `environment` value has been passed, run both `std` and `no_std` checks.

## Continuous Deployment

### Publish crates

Compile `scripts/publish.rs` using this command:

```
rustc scripts/publish.rs --crate-type bin --out-dir scripts
```

Run `scripts/publish` using this command

```
./scripts/publish crate_name
```

where `crate_name` is the name of the crate to publish


## Disclaimer

Burn is currently in active development, and there will be breaking changes. While any resulting
issues are likely to be easy to fix, there are no guarantees at this stage.

## Sponsors

You can sponsor the founder of Burn from his
[GitHub Sponsors profile](https://github.com/sponsors/nathanielsimard). The Burn-rs organization
doesn't yet have a fiscal entity, but other sponsor methods might become available as the project
grows.

Thanks to all current sponsors 🙏.

<a href="https://github.com/smallstepman"><img src="https://github.com/smallstepman.png" width="60px" style="border-radius: 50%;" alt="smallstepman" /></a>
<a href="https://github.com/premAI-io"><img src="https://github.com/premAI-io.png" width="60px" style="border-radius: 50%;" alt="premAI-io" /></a>

## License

Burn is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
See [LICENSE-APACHE](./LICENSE-APACHE) and [LICENSE-MIT](./LICENSE-MIT) for details. Opening a pull
request is assumed to signal agreement with these licensing terms.
