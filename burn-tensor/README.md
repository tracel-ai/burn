# Burn Tensor

> Burn Tensor Library

This library provides multiple tensor implementations hidden behind an easy to use API that supports reverse mode automatic differentiation.

## Features

* Flexible ✨
* CPU + GPU 🙏
* Multi-Threads 🚀
* Intuitive Usage 😌
* No Global State 🚫
* Multiple Backends 🦾
* Reverse Mode Autodiff 🔥

### Backends

For now, only two backends are implementated, but adding new ones should be easy.

* [X] Pytorch using [tch-rs](https://github.com/LaurentMazare/tch-rs)
* [X] 100% Rust backend using [ndarray](https://github.com/rust-ndarray/ndarray)
* [ ] Tensorflow using [tensorflow-rust](https://github.com/tensorflow/rust)
* [ ] ...

## Usage

### Basic

This library separates data from tensors, where the former is used to create new tensors and serialize/deserialize data, and the later is used to execute tensor operations.

```rust
    use burn_tensor::tensor::*;

    let data_x = Data::<f32, 3>::random(Shape::new([32, 24, 24]), Distribution::Standard);
    let data_y = Data::<f32, 3>::random(Shape::new([32, 24, 24]), Distribution::Standard);
```

Tensors can be created from the generated data.

```rust
    use burn_tensor::tensor::backend::ndarray::*;
    use burn_tensor::tensor::backend::tch::*;

    let x_ndarray = NdArrayTensor::from_data(data_x.clone());
    let x_tch = TchTensor::from_data(data_x, TchDevice::Cpu);

    let y_ndarray = NdArrayTensor::from_data(data_y.clone());
    let y_tch = TchTensor::from_data(data_y, TchDevice::Cpu);
```

Operations can be executed only with other tensors of the same type.

```rust
    use burn_tensor::tensor::ops::*;

    let z_ndarray = x_ndarray.matmul(&y_ndarray);
    let z_tch = x_tch.matmul(&y_tch);
```

Tensors can be exported to Data for easy serialization.

```rust
    let z_ndarray_data = z_ndarray.into_data();
    let z_tch_data = z_tch.into_data();

    assert_eq!(z_ndarray_data, z_tch_data);
```

### Autodiff

Automatic differentiation is implemented as just another tensor backend without any global state.
It's possible since we keep track of the order in which each operation as been executed and the tape is only created when calculating the gradients.
To do so, each operation creates a new node which has a reference to its parent nodes.
Therefore, creating the tape only requires a simple and efficent graph traversal algorithm.

```rust
    use burn_tensor::tensor::backend::autodiff::*;

    let x = ADTensor::from_tensor(x_ndarray);
    let y = ADTensor::from_tensor(y_ndarray);

    let z = x.matmul(&y);

    let grads = z.backward();

    let x_grad = grads.wrt(&x);
    let y_grad = grads.wrt(&y);
```

## Note

This crate can be use alone without the entire burn stack and with only selected backends for smaller binaries.
