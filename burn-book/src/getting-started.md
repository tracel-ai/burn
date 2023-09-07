# Getting Started

## Installing Rust

Burn is a deep learning framework in the Rust programming language. Therefore, it goes without
saying that one must have basic notions of Rust. Reading the first chapters of the
[Rust book](https://doc.rust-lang.org/book/) is a great way to begin.

In particular, the books'
[installation page](https://doc.rust-lang.org/book/ch01-01-installation.html) explains in details
the most convenient way for you to install Rust on your computer, which is the very first thing to
do in order to run Burn.

## Creating a Burn application

Once Rust is correctly installed, create a new Rust application by using Rust's package manager
Cargo, which was installed with Rust. In the directory of your choice, run

```console
cargo new my_burn_app
```

This will create the `my_burn_app` project directory. Head inside and open the `Cargo.toml` file. It
should contain something like:

```toml
[package]
name = "my_burn_app"
version = "0.1.0"
edition = "2021"

# See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

[dependencies]
```

Under dependencies, add

```toml
burn = { version = "0.9.0", features = ["wgpu"] }
```

Then, to compile the dependencies, execute

```console
cargo build
```

This will install Burn, along with the WGPU backend for Burn, which allows to execute low-level
operations on every platform, using the GPU.

## Writing a code snippet

Now open `src/main.rs` and replace its content with

```rust, ignore
use burn::tensor::Tensor;
use burn::backend::WgpuBackend;

// Type alias for the backend to use.
type Backend = WgpuBackend;

fn main() {
    // Creation of two tensors, the first with explicit values and the second one with ones, with the same shape as the first
    let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]]);
    let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);

    // Print the element-wise addition (done with the WGPU backend) of the two tensors.
    println!("{}", tensor_1 + tensor_2);
}
```

By running `cargo run`, you should now see the result of the addition:

```console
Tensor {
  data: [[3.0, 4.0], [5.0, 6.0]],
  shape:  [2, 2],
  device:  BestAvailable,
  backend:  "wgpu",
  kind:  "Float",
  dtype:  "f32",
}
```

While the previous example is somewhat trivial, the upcoming
[basic workflow section](./basic-workflow/README.md) will walk you through a much more relevant
example for deep learning applications.
