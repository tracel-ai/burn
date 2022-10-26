<div align="center">
<img src="./assets/logo-burn-full.png" width="200px"/>

[![Current Crates.io Version](https://img.shields.io/crates/v/burn.svg)](https://crates.io/crates/burn)
[![Test Status](https://github.com/burn-rs/burn/actions/workflows/test-burn.yml/badge.svg)](https://github.com/burn-rs/burn/actions/workflows/test-burn.yml)
[![Documentation](https://docs.rs/burn/badge.svg)](https://docs.rs/burn)
[![Rust Version](https://img.shields.io/badge/Rust-1.65.0-blue)](https://releases.rs/docs/unreleased/1.65.0)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/burn-rs/burn/blob/master/LICENSE)

> This library aims to be a complete deep learning framework with extreme flexibility written in Rust. 
> The goal would be to satisfy researchers as well as practitioners making it easier to experiment, train and deploy your models.

<div align="left">

__Sections__

* [Features](#features)
* [Get Started](#get-started)
    * [Examples](#examples)
        * [MNIST](#mnist)
    * [Components](#components)
        * [Backend](#backend)
        * [Tensor](#tensor)
        * [Module](#module)
        * [Forward](#forward)
        * [Config](#config)
        * [Learner](#learner)
* [License](#license)

## Features

 * Flexible and intuitive custom neural network module 🤖
 * Stateless and thread safe forward pass 🚀
 * Fast training with full support for `metric`, `logging` and `checkpoining` 🌟
 * [Burn-Tensor](https://github.com/burn-rs/burn/tree/doc/readme/burn-tensor): Tensor library with autodiff, CPU and GPU support 🔥
 * [Burn-Dataset](https://github.com/burn-rs/burn/tree/doc/readme/burn-dataset): Dataset library with multiple utilities and sources 📚

## Get Started

The best way to get started with burn is the look at the [examples](#examples).
Also, this may be a good idea to checkout the main [components](#components) to get a quick overview of how to use burn.

### Examples

For now there is only one example, but more to come 💪.

#### MNIST

The [MNIST](https://github.com/burn-rs/burn/blob/main/examples/mnist) example is not just of small script that shows you how to train a basic model, but it's a quick one showing you how to:

* Define your own custom [module](#module) (MLP).
* Create the data pipeline from a raw dataset to a batched multi-threaded fast DataLoader.
* Configure a [learner](#learner) to display and log metrics as well as to keep training checkpoints.

### Components

Knowing the main components will be of great help when starting playing with `burn`.

#### Backend

Almost everything is based on the `Backend` trait, which allows to run tensor operations with different implementations without having to change your code.
A backend does not necessary have autodiff capabilities, therefore you can use `ADBackend` when you require it.

#### Tensor

The `Tensor` struct is at the core of the `burn` framework.
It takes two generic parameters, the `Backend` and the number of dimensions `D`,

```rust
use burn::tensor::{Tensor, Shape, Data};
use burn::tensor::backend::{Backend, NdArrayBackend, TchBackend};

fn my_func<B: Backend>() {
    let _my_tensor = Tensor::<B, 2>::ones(Shape::new([3, 3]));
}

fn main() {
    my_func<NdArrayBackend<f32>>();
    my_func<TchBackend<f32>>();
}
```

#### Module

The `Module` derive let your create your own neural network module similar to PyTorch.

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

#### Forward

The `Forward` trait can also be implemented by your module.

```rust
use burn::module::Forward;
use burn::tensor::Tensor;

impl<B: Backend> Forward<Tensor<B, 2>, Tensor<B, 2>> for MyModule<B> {
   fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
       let mut x = input;

       for _ in 0..self.repeat {
           x = self.my_param.forward(x);
       }

       x
   }
}
```

Note that you can implement multiple time the `Forward` trait with different inputs and outputs.

#### Config

The `Config` derive let you define serializable and deserializable configurations or hyper-parameters for your [modules](#module) or any components.

```rust
use burn::config::Config;

#[derive(Config)]
struct MyConfig {
    #[config(default = 1.0e-6)]
    pub epsilone: usize,
    pub dim: usize,
}
```
The derive also add usefull methods to your config.

```rust
fn my_func() {
    let config = MyConfig::new(100);
    println!("{}", config.epsilone); // 1.0.e-6
    println!("{}", config.dim); // 100
    let config =  MyConfig::new(100).with_epsilone(1.0e-8);
    println!("{}", config.epsilone); // 1.0.e-8
}
```

#### Learner

The `Learner` is the main `struct` that let you train a neural network with support for `logging`, `metric`, `checkpointing` and more.
In order to create a learner, you must use the `LearnerBuilder`.

```rust
use burn::train::LearnerBuilder;

let learner = LearnerBuilder::new("/tmp/artifact_dir")
    .metric_train_plot(AccuracyMetric::new())
    .metric_valid_plot(AccuracyMetric::new())
    .metric_train(LossMetric::new())
    .metric_valid(LossMetric::new())
    .with_file_checkpointer::<f32>(2)
    .num_epochs(config.num_epochs)
    .build(model, optim);
```

See this [example](./burn/examples/mnist.rs) for a real usage.

## License

Burn is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
See [LICENSE-APACHE](./LICENSE-APACHE) and [LICENSE-MIT](./LICENSE-MIT) for details.
Opening a pull request is assumed to signal agreement with these licensing terms.
