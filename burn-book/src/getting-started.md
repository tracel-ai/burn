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

This will create the `my_burn_app` project directory. Head inside with

```console
cd my_burn_app
```

then add the dependency with

```console
cargo add burn --features wgpu
```

and compile it by executing

```console
cargo build
```

This will install Burn, along with the WGPU backend for Burn, which allows to execute low-level
operations on every platform, using the GPU.

## Writing a code snippet

Now open `src/main.rs` and replace its content with

```rust, ignore
use burn::tensor::Tensor;
use burn::backend::Wgpu;

// Type alias for the backend to use.
type Backend = Wgpu;

fn main() {
    // Creation of two tensors, the first with explicit values and the second one with ones, with the same shape as the first
    let tensor_1 = Tensor::<Backend, 2>::from_data_devauto([[2., 3.], [4., 5.]]);
    let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);

    // Print the element-wise addition (done with the WGPU backend) of the two tensors.
    println!("{}", tensor_1 + tensor_2);
}
```

By running `cargo run`, you should now see the result of the addition:

```console
Tensor {
  data:
[[3.0, 4.0],
 [5.0, 6.0]],
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

## Running examples

Burn uses a [Python library by HuggingFace](https://huggingface.co/docs/datasets/index) to download
datasets. Therefore, in order to run examples, you will need to install Python. Follow the
instructions on the [official website](https://www.python.org/downloads/) to install Python on your
computer.

Many Burn examples are available in the [examples](https://github.com/tracel-ai/burn/tree/main/examples)
directory. 
To run one, please refer to the example's README.md for the specific command to
execute.
