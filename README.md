<div align="center">
<img src="https://github.com/burn-rs/burn/blob/main/assets/logo-burn-full.png" width="200px"/>

[![Current Crates.io Version](https://img.shields.io/crates/v/burn.svg)](https://crates.io/crates/burn)
[![Test Status](https://github.com/burn-rs/burn/actions/workflows/test-burn.yml/badge.svg)](https://github.com/burn-rs/burn/actions/workflows/test-burn.yml)
[![Documentation](https://docs.rs/burn/badge.svg)](https://docs.rs/burn)
[![Rust Version](https://img.shields.io/badge/Rust-1.65.0-blue)](https://releases.rs/docs/unreleased/1.65.0)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/burn-rs/burn/blob/master/LICENSE)

> This library aims to be a complete deep learning framework with extreme flexibility written in Rust. 
> The goal would be to satisfy researchers as well as practitioners making it easier to experiment, train and deploy your models.

<div align="left">

## Features

 * Flexible and intuitive custom neural network module 🤖
 * Stateless and thread safe forward pass 🚀
 * Fast training with full support for `metric`, `logging` and `checkpoining` 🌟
 * [Burn-Tensor](https://github.com/burn-rs/burn/burn-tensor): Tensor library with autodiff, CPU and GPU support 🔥
 * [Burn-Dataset](https://github.com/burn-rs/burn/burn-dataset): Dataset library with multiple utilities and sources 📚

## Details

### Example

Full example showing most of the features from `burn` [MNIST](https://github.com/burn-rs/burn/blob/main/burn/examples/mnist.rs).

### Components

Knowing the main components will be of great help when starting playing with `burn`.

#### __Backend__

Almost everything is based on the `Backend` trait, which allows to run tensor operations with different implementations without having to change your code.
A backend does not necessary have autodiff capabilities, therefore you can use `ADBackend` when you need it.

#### Tensor

The `Tensor` struct is at the core of the `burn` framework.
It takes two generic parameters, the `Backend` and the number of dimensions `D`,

```rust
use burn::tensor::{Tensor, Shape, Data};
use burn::tensor::backend::{NdArrayBackend, TchBackend};

let my_ndarray_matrix = Tensor::<NdArrayBackend<f32>, 2>::ones(Shape::new([3, 3]));
let my_tch_matrix = Tensor::<TchBackend<f32>, 2>::from_data(
    Data::from([[1.0, 7.0], [13.0, -3.0]])
);
```

Note that `Data` is not specific to any backend.

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
