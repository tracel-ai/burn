<div align="center">
<img src="https://raw.githubusercontent.com/burn-rs/burn/main/assets/logo-burn-full.png" width="200px"/>

[![Discord](https://img.shields.io/discord/1038839012602941528.svg?color=7289da&&logo=discord)](https://discord.gg/KcVGzmCcWj)
[![Test Status](https://github.com/burn-rs/burn/actions/workflows/test.yml/badge.svg)](https://github.com/burn-rs/burn/actions/workflows/test.yml)
[![Documentation](https://docs.rs/burn/badge.svg)](https://docs.rs/burn)
[![Current Crates.io Version](https://img.shields.io/crates/v/burn.svg)](https://crates.io/crates/burn)
[![Rust Version](https://img.shields.io/badge/Rust-1.65.0+-blue)](https://releases.rs/docs/released/1.65.0)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/burn-rs/burn/blob/master/LICENSE)

> This library aims to be a complete deep learning framework with extreme flexibility written in Rust.
> The goal would be to satisfy researchers as well as practitioners making it easier to experiment, train and deploy your models.

<div align="left">

*__Disclamer__* _Burn is currently in active development, and there will be breaking changes. While any resulting issues are likely to be easy to fix, there are no guarantees at this stage._

__Sections__

* [Features](#features)
* [Get Started](#get-started)
    * [Examples](#examples)
    * [Components](#components)
        * [Backend](#backend)
        * [Tensor](#tensor)
        * [Module](#module)
        * [Config](#config)
        * [Learner](#learner)
* [License](#license)

## Features

 * Flexible and intuitive custom neural network [module](#module) 🔥
 * [Training](#learner) with full support for `metric`, `logging` and `checkpointing` 📈
 * [Tensor](#tensor) crate with backends as pluging 🔧
   * [Tch](https://github.com/burn-rs/burn/tree/main/burn-tch) backend with CPU/GPU support 🚀
   * [NdArray](https://github.com/burn-rs/burn/tree/main/burn-ndarray) backend with fast compile time 👌
   * [Autodiff](https://github.com/burn-rs/burn/tree/main/burn-autodiff) backend making any backend differentiable 🌟
 * [Dataset](https://github.com/burn-rs/burn/tree/main/burn-dataset) crate with multiple utilities and sources 📚

## Get Started

The best way to get started with `burn` is to clone the repo and play with the [examples](#examples).
This may also be a good idea to take a look the main [components](#components) of `burn` to get a quick overview of the fundamental building blocks.

### Examples

* [MNIST](https://github.com/burn-rs/burn/tree/main/examples/mnist) train a model on CPU/GPU using different backends.
* [Text Classification](https://github.com/burn-rs/burn/tree/main/examples/text-classification) train a transformer encoder from scratch on GPU.

### Components

Knowing the main components will be of great help when starting playing with `burn`.

#### Backend

Almost everything is based on the `Backend` trait, which allows to run tensor operations with different implementations without having to change your code.
A backend does not necessary have autodiff capabilities, the `ADBackend` trait is there to specify when autodiff is required.

#### Tensor

The `Tensor` struct is at the core of the `burn` framework.
It takes two generic parameters, the `Backend` and the number of dimensions `D`,

Backpropagation is also supported on any backend by making them auto differentiable using a simple decorator.

```rust
use burn::tensor::backend::{ADBackend, Backend};
use burn::tensor::{Distribution, Tensor};
use burn_autodiff::ADBackendDecorator;
use burn_ndarray::NdArrayBackend;
use burn_tch::TchBackend;

fn simple_function<B: Backend>() -> Tensor<B, 2> {
    let x = Tensor::<B, 2>::random([3, 3], Distribution::Standard);
    let y = Tensor::<B, 2>::random([3, 3], Distribution::Standard);

    x.matmul(&y)
}

fn simple_function_grads<B: ADBackend>() -> B::Gradients {
    let z = simple_function::<B>();

    z.backward()
}

fn main() {
    let _z = simple_function::<NdArrayBackend<f32>>(); // Compiles
    let _z = simple_function::<TchBackend<f32>>(); // Compiles

    let _grads = simple_function_grads::<NdArrayBackend<f32>>(); // Doesn't compile
    let _grads = simple_function_grads::<TchBackend<f32>>(); // Doesn't compile

    type ADNdArrayBackend = ADBackendDecorator<NdArrayBackend<f32>>;
    type ADTchBackend = ADBackendDecorator<TchBackend<f32>>;

    let _grads = simple_function_grads::<ADNdArrayBackend>(); // Compiles
    let _grads = simple_function_grads::<ADTchBackend>(); // Compiles
}
```

#### Module

The `Module` derive let your create your own neural network modules similar to PyTorch.

```rust
use burn::nn;
use burn::module::{Param, Module};
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
struct MyModule<B: Backend> {
  my_param: Param<nn::Linear<B>>,
  repeat: usize,
}
```

Note that only the fields wrapped inside `Param` are updated during training, and the other ones should implement `Clone`.

#### Config

The `Config` derive lets you define serializable and deserializable configurations or hyper-parameters for your [modules](#module) or any components.

```rust
use burn::config::Config;

#[derive(Config)]
struct MyConfig {
    #[config(default = 1.0e-6)]
    pub epsilon: usize,
    pub dim: usize,
}
```
The derive also adds useful methods to your config.

```rust
fn main() {
    let config = MyConfig::new(100);
    println!("{}", config.epsilon); // 1.0.e-6
    println!("{}", config.dim); // 100
    let config =  MyConfig::new(100).with_epsilon(1.0e-8);
    println!("{}", config.epsilon); // 1.0.e-8
}
```

#### Learner

The `Learner` is the main `struct` that let you train a neural network with support for `logging`, `metric`, `checkpointing` and more.
In order to create a learner, you must use the `LearnerBuilder`.

```rust
use burn::train::LearnerBuilder;
use burn::train::metric::{AccuracyMetric, LossMetric};

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
        .with_file_checkpointer::<f32>(2)
        .num_epochs(10)
        .build(model, optim);

    let _model_trained = learner.fit(dataloader_train, dataloader_valid);
}
```

See this [example](https://github.com/burn-rs/burn/tree/main/examples/mnist) for a real usage.

## License

Burn is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
See [LICENSE-APACHE](./LICENSE-APACHE) and [LICENSE-MIT](./LICENSE-MIT) for details.
Opening a pull request is assumed to signal agreement with these licensing terms.
