# Backend

We have effectively written most of the necessary code to train our model. However, we have not
explicitly designated the backend to be used at any point. This will be defined in the main
entrypoint of our program, namely the `main` function defined in `src/main.rs`.

```rust , ignore
use burn::optim::AdamConfig;
use burn::backend::{Autodiff, Wgpu, wgpu::AutoGraphicsApi};
use guide::model::ModelConfig;

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    guide::training::train::<MyAutodiffBackend>(
        "/tmp/guide",
        guide::training::TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device,
    );
}
```

<details>
<summary><strong>🦀 Packages, Crates and Modules</strong></summary>

You might be wondering why we use the `guide` prefix to bring the different modules we just
implemented into scope. Instead of including the code in the current guide in a single file, we
separated it into different files which group related code into _modules_. The `guide` is simply the
name we gave to our _crate_, which contains the different files. Below is a brief explanation of the
different parts of the Rust module system.

A **package** is a bundle of one or more crates that provides a set of functionality. A package
contains a `Cargo.toml` file that describes how to build those crates. Burn is a package.

A **crate** is a compilation unit in Rust. It could be a single file, but it is often easier to
split up crates into multiple _modules_ and possibly multiple files. A crate can come in one of two
forms: a binary crate or a library crate. When compiling a crate, the compiler first looks in the
crate root file (usually `src/lib.rs` for a library crate or `src/main.rs` for a binary crate). Any
module declared in the crate root file will be inserted in the crate for compilation. For this demo
example, we will define a library crate where all the individual modules (model, data, training,
etc.) are listed inside `src/lib.rs` as follows:

```
pub mod data;
pub mod inference;
pub mod model;
pub mod training;
```

A **module** lets us organize code within a crate for readability and easy reuse. Modules also allow
us to control the _privacy_ of items. The `pub` keyword used above, for example, is employed to make
a module publicly available inside the crate.

The entry point of our program is the `main` function, defined in the `examples/guide.rs` file. The
file structure for this example is illustrated below:

```
guide
├── Cargo.toml
├── examples
│   └── guide.rs
└── src
    ├── data.rs
    ├── inference.rs
    ├── lib.rs
    ├── model.rs
    └── training.rs
```

The source for this guide can be found in our
[GitHub repository](https://github.com/tracel-ai/burn/tree/main/examples/guide) which can be used to
run this basic workflow example end-to-end.\

</details><br>

In this example, we use the `Wgpu` backend which is compatible with any operating system and will
use the GPU. For other options, see the Burn README. This backend type takes the graphics api, the
float type and the int type as generic arguments that will be used during the training. By leaving
the graphics API as `AutoGraphicsApi`, it should automatically use an API available on your machine.
The autodiff backend is simply the same backend, wrapped within the `Autodiff` struct which imparts
differentiability to any backend.

We call the `train` function defined earlier with a directory for artifacts, the configuration of
the model (the number of digit classes is 10 and the hidden dimension is 512), the optimizer
configuration which in our case will be the default Adam configuration, and the device which can be
obtained from the backend.

When running the example, we can see the training progression through a basic CLI dashboard:

<img title="a title" alt="Alt text" src="./training-output.png">
