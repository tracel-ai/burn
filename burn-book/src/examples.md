# Examples

In the [next chapter](./basic-workflow) you'll have the opportunity to implement the whole Burn
`guide` example yourself in a step by step manner.

Many additional Burn examples are available in the
[examples](https://github.com/tracel-ai/burn/tree/main/examples) directory. Burn examples are
organized as library crates with one or more examples that are executable binaries. An example can
then be executed using the following cargo command line in the root of the Burn repository:

```bash
cargo run --example <example name>
```

To learn more about crates and examples, read the Rust section below.

<details>
<summary><strong>🦀 About Rust crates</strong></summary>

Each Burn example is a **package** which are subdirectories of the `examples` directory. A package
is composed of one or more **crates**.

A package is a bundle of one or more crates that provides a set of functionality. A package contains
a `Cargo.toml` file that describes how to build those crates.

A crate is a compilation unit in Rust. It could be a single file, but it is often easier to split up
crates into multiple **modules**.

A module lets us organize code within a crate for readability and easy reuse. Modules also allow us
to control the _privacy_ of items. For instance the `pub(crate)` keyword is employed to make a
module publicly available inside the crate. In the snippet below there are four modules declared,
two of them are public and visible to the users of the crates, one of them is public inside the
crate only and crate users cannot see it, at last one is private when there is no keyword. These
modules can be single files or a directory with a `mod.rs` file inside.

```rust, ignore
pub mod data;
pub mod inference;
pub(crate) mod model;
mod training;
```

A crate can come in one of two forms: a **binary crate** or a **library crate**. When compiling a
crate, the compiler first looks in the crate root file (`src/lib.rs` for a library crate and
`src/main.rs` for a binary crate). Any module declared in the crate root file will be inserted in
the crate for compilation.

All Burn examples are library crates and they can contain one or more executable examples that uses
the library. We even have some Burn examples that uses the library crate of other examples.

The examples are unique files under the `examples` directory. Each file produces an executable file
with the same name. Each example can then be executed with `cargo run --example <executable name>`.

Below is a file tree of a typical Burn example package:

```
examples/burn-example
├── Cargo.toml
├── examples
│   ├── example1.rs      ---> compiled to example1 binary
│   ├── example2.rs      ---> compiled to example2 binary
│   └── ...
└── src
    ├── lib.rs           ---> this is the root file for a library
    ├── module1.rs
    ├── module2.rs
    └── ...
```

</details><br>

The following additional examples are currently available if you want to check them out:

| Example                                                                                                   | Description                                                                                                                                                                                  |
| :-------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Custom CSV Dataset](https://github.com/tracel-ai/burn/tree/main/examples/custom-csv-dataset)             | Implements a dataset to parse CSV data for a regression task.                                                                                                                                |
| [Regression](https://github.com/tracel-ai/burn/tree/main/examples/simple-regression)                      | Trains a simple MLP on the CSV dataset for the regression task.                                                                                                                              |
| [Custom Image Dataset](https://github.com/tracel-ai/burn/tree/main/examples/custom-image-dataset)         | Trains a simple CNN on custom image dataset following a simple folder structure.                                                                                                             |
| [Custom Renderer](https://github.com/tracel-ai/burn/tree/main/examples/custom-renderer)                   | Implements a custom renderer to display the [`Learner`](./building-blocks/learner.md) progress.                                                                                              |
| [Image Classification Web](https://github.com/tracel-ai/burn/tree/main/examples/image-classification-web) | Image classification web browser demo using Burn, WGPU and WebAssembly.                                                                                                                      |
| [MNIST Inference on Web](https://github.com/tracel-ai/burn/tree/main/examples/mnist-inference-web)        | An interactive MNIST inference demo in the browser. The demo is available [online](https://burn.dev/demo/).                                                                                  |
| [MNIST Training](https://github.com/tracel-ai/burn/tree/main/examples/mnist)                              | Demonstrates how to train a custom [`Module`](./building-blocks/module.md) (MLP) with the [`Learner`](./building-blocks/learner.md) configured to log metrics and keep training checkpoints. |
| [Named Tensor](https://github.com/tracel-ai/burn/tree/main/examples/named-tensor)                         | Performs operations with the experimental `NamedTensor` feature.                                                                                                                             |
| [ONNX Import Inference](https://github.com/tracel-ai/burn/tree/main/examples/onnx-inference)              | Imports an ONNX model pre-trained on MNIST to perform inference on a sample image with Burn.                                                                                                 |
| [PyTorch Import Inference](https://github.com/tracel-ai/burn/tree/main/examples/pytorch-import)           | Imports a PyTorch model pre-trained on MNIST to perform inference on a sample image with Burn.                                                                                               |
| [Text Classification](https://github.com/tracel-ai/burn/tree/main/examples/text-classification)           | Trains a text classification transformer model on the AG News or DbPedia datasets. The trained model can then be used to classify a text sample.                                              |
| [Text Generation](https://github.com/tracel-ai/burn/tree/main/examples/text-generation)                   | Trains a text generation transformer model on the DbPedia dataset.                                                                                                                           |

For more information on each example, see their respective `README.md` file. Be sure to check out
the [examples](https://github.com/tracel-ai/burn/tree/main/examples) directory for an up-to-date
list.

<div class="warning">

Note that some examples use the
[`datasets` library by HuggingFace](https://huggingface.co/docs/datasets/index) to download the
datasets required in the examples. This is a Python library, which means that you will need to
install Python before running these examples. This requirement will be clearly indicated in the
example's README when applicable.

</div>
