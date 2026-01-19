# Model

The first step is to create a project and add the different Burn dependencies. Start by creating a
new project with Cargo:

```console
cargo new guide
```

As [mentioned previously](../getting-started.md#creating-a-burn-application), this will initialize
your `guide` project directory with a `Cargo.toml` and a `src/main.rs` file.

In the `Cargo.toml` file, add the `burn` dependency with `train`, `vision` and `wgpu` features.
Since we disable the default features, we also want to enable `std`, `tui` (for the dashboard) and
`fusion` for wgpu. Then run `cargo build` to build the project and import all the dependencies.

```toml
[package]
name = "guide"
version = "0.1.0"
edition = "2024"

[dependencies]
# Disable autotune default for convolutions
burn = { version = "~0.21", features = ["std", "tui", "train", "vision", "wgpu", "fusion"], default-features = false }
# burn = { version = "~0.21", features = ["train", "vision", "wgpu"] }
```

Our goal will be to create a basic convolutional neural network used for image classification. We
will keep the model simple by using two convolution layers followed by two linear layers, some
pooling and ReLU activations. We will also use dropout to improve training performance.

Let us start by defining our model struct in a new file `src/model.rs`.

```rust , ignore
use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}
```

There are two major things going on in this code sample.

1. You can create a deep learning module with the `#[derive(Module)]` attribute on top of a struct.
   This will generate the necessary code so that the struct implements the `Module` trait. This
   trait will make your module both trainable and (de)serializable while adding related
   functionalities. Like other attributes often used in Rust, such as `Clone`, `PartialEq` or
   `Debug`, each field within the struct must also implement the `Module` trait.

   <details>
   <summary><strong>🦀 Trait</strong></summary>

   Traits are a powerful and flexible Rust language feature. They provide a way to define shared
   behavior for a particular type, which can be shared with other types.

   A type's behavior consists of the methods called on that type. Since all `Module`s should
   implement the same functionality, it is defined as a trait. Implementing a trait on a particular
   type usually requires the user to implement the defined behaviors of the trait for their types,
   though that is not the case here as explained above with the `derive` attribute. Check out the
   [explainer below](#derive-attribute) to learn why.

   For more details on traits, take a look at the
   [associated chapter](https://doc.rust-lang.org/book/ch10-02-traits.html) in the Rust Book.
   </details><br>

   <details id="derive-attribute">
   <summary><strong>🦀 Derive Macro</strong></summary>

   The `derive` attribute allows traits to be implemented easily by generating code that will
   implement a trait with its own default implementation on the type that was annotated with the
   `derive` syntax.

   This is accomplished through a feature of Rust called
   [procedural macros](https://doc.rust-lang.org/reference/procedural-macros.html), which allow us
   to run code at compile time that operates over Rust syntax, both consuming and producing Rust
   syntax. Using the attribute `#[my_macro]`, you can effectively extend the provided code. You will
   see that the derive macro is very frequently employed to recursively implement traits, where the
   implementation consists of the composition of all fields.

   In this example, we want to derive the [`Module`](../building-blocks/module.md) and `Debug`
   traits.

   ```rust, ignore
   #[derive(Module, Debug)]
   pub struct MyCustomModule<B: Backend> {
       linear1: Linear<B>,
       linear2: Linear<B>,
       activation: Relu,
   }
   ```

   The basic `Debug` implementation is provided by the compiler to format a value using the `{:?}`
   formatter. For ease of use, the `Module` trait implementation is automatically handled by Burn so
   you don't have to do anything special. It essentially acts as parameter container.

   For more details on derivable traits, take a look at the Rust
   [appendix](https://doc.rust-lang.org/book/appendix-03-derivable-traits.html),
   [reference](https://doc.rust-lang.org/reference/attributes/derive.html) or
   [example](https://doc.rust-lang.org/rust-by-example/trait/derive.html).
   </details><br>

2. Note that the struct is generic over the [`Backend`](../building-blocks/backend.md) trait. The
   backend trait abstracts the underlying low level implementations of tensor operations, allowing
   your new model to run on any backend. Contrary to other frameworks, the backend abstraction isn't
   determined by a compilation flag or a device type. This is important because you can extend the
   functionalities of a specific backend (see
   [backend extension section](../advanced/backend-extension)), and it allows for an innovative
   [autodiff system](../building-blocks/autodiff.md). You can also change backend during runtime,
   for instance to compute training metrics on a cpu backend while using a gpu one only to train the
   model. In our example, the backend in use will be determined later on.

   <details>
   <summary><strong>🦀 Trait Bounds</strong></summary>

   Trait bounds provide a way for generic items to restrict which types are used as their
   parameters. The trait bounds stipulate what functionality a type implements. Therefore, bounding
   restricts the generic to types that conform to the bounds. It also allows generic instances to
   access the methods of traits specified in the bounds.

   For a simple but concrete example, check out the
   [Rust By Example on bounds](https://doc.rust-lang.org/rust-by-example/generics/bounds.html).

   In Burn, the `Backend` trait enables you to run tensor operations using different implementations
   as it abstracts tensor, device and element types. The
   [getting started example](../getting-started.md#writing-a-code-snippet) illustrates the advantage
   of having a simple API that works for different backend implementations. While it used the WGPU
   backend, you could easily swap it with any other supported backend.

   ```rust, ignore
   // Choose from any of the supported backends.
   // type Backend = Candle<f32, i64>;
   // type Backend = LibTorch<f32>;
   // type Backend = NdArray<f32>;
   type Backend = Wgpu;

   // Creation of two tensors.
   let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]], &device);
   let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);

   // Print the element-wise addition (done with the selected backend) of the two tensors.
   println!("{}", tensor_1 + tensor_2);
   ```

   For more details on trait bounds, check out the Rust
   [trait bound section](https://doc.rust-lang.org/book/ch10-02-traits.html#trait-bound-syntax) or
   [reference](https://doc.rust-lang.org/reference/items/traits.html#trait-bounds).

   </details><br>

Note that each time you create a new file in the `src` directory you also need to explicitly add
this module to the `main.rs` file. For instance after creating the `model.rs`, you need to add the
following at the top of the main file:

```rust , ignore
mod model;
#
# fn main() {
# }
```

Next, we need to instantiate the model for training.

```rust , ignore
# use burn::{
#     nn::{
#         conv::{Conv2d, Conv2dConfig},
#         pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
#         Dropout, DropoutConfig, Linear, LinearConfig, Relu,
#     },
#     prelude::*,
# };
#
# #[derive(Module, Debug)]
# pub struct Model<B: Backend> {
#     conv1: Conv2d<B>,
#     conv2: Conv2d<B>,
#     pool: AdaptiveAvgPool2d,
#     dropout: Dropout,
#     linear1: Linear<B>,
#     linear2: Linear<B>,
#     activation: Relu,
# }
#
#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}
```

At a glance, you can view the model configuration by printing the model instance:

```rust , ignore
#![recursion_limit = "256"]
mod model;

use crate::model::ModelConfig;
use burn::backend::Wgpu;

fn main() {
    type MyBackend = Wgpu<f32, i32>;

    let device = Default::default();
    let model = ModelConfig::new(10, 512).init::<MyBackend>(&device);

    println!("{model}");
}
```

Output:

```rust , ignore
Model {
  conv1: Conv2d {ch_in: 1, ch_out: 8, stride: [1, 1], kernel_size: [3, 3], dilation: [1, 1], groups: 1, padding: Valid, params: 80}
  conv2: Conv2d {ch_in: 8, ch_out: 16, stride: [1, 1], kernel_size: [3, 3], dilation: [1, 1], groups: 1, padding: Valid, params: 1168}
  pool: AdaptiveAvgPool2d {output_size: [8, 8]}
  dropout: Dropout {prob: 0.5}
  linear1: Linear {d_input: 1024, d_output: 512, bias: true, params: 524800}
  linear2: Linear {d_input: 512, d_output: 10, bias: true, params: 5130}
  activation: Relu
  params: 531178
}
```

<details>
<summary><strong>🦀 References</strong></summary>

In the previous example, the `init()` method signature uses `&` to indicate that the parameter types
are references: `&self`, a reference to the current receiver (`ModelConfig`), and
`device: &B::Device`, a reference to the backend device.

```rust, ignore
pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
    Model {
        // ...
    }
}
```

References in Rust allow us to point to a resource to access its data without owning it. The idea of
ownership is quite core to Rust and is worth
[reading up on](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html).

In a language like C, memory management is explicit and up to the programmer, which means it is easy
to make mistakes. In a language like Java or Python, memory management is automatic with the help of
a garbage collector. This is very safe and straightforward, but also incurs a runtime cost.

In Rust, memory management is rather unique. Aside from simple types that implement
[`Copy`](https://doc.rust-lang.org/std/marker/trait.Copy.html) (e.g.,
[primitives](https://doc.rust-lang.org/rust-by-example/primitives.html) like integers, floats,
booleans and `char`), every value is _owned_ by some variable called the _owner_. Ownership can be
transferred from one variable to another and sometimes a value can be _borrowed_. Once the _owner_
variable goes out of scope, the value is _dropped_, which means that any memory it allocated can be
freed safely.

Because the method does not own the `self` and `device` variables, the values the references point
to will not be dropped when the reference stops being used (i.e., the scope of the method).

For more information on references and borrowing, be sure to read the
[corresponding chapter](https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html) in the
Rust Book.

</details><br>

When creating a custom neural network module, it is often a good idea to create a config alongside
the model struct. This allows you to define default values for your network, thanks to the `Config`
attribute. The benefit of this attribute is that it makes the configuration serializable, enabling
you to painlessly save your model hyperparameters, enhancing your experimentation process. Note that
a constructor will automatically be generated for your configuration, which will take in as input
values the parameters which do not have default values:
`let config = ModelConfig::new(num_classes, hidden_size);`. The default values can be overridden
easily with builder-like methods: (e.g `config.with_dropout(0.2);`)

The first implementation block is related to the initialization method. As we can see, all fields
are set using the configuration of the corresponding neural network's underlying module. In this
specific case, we have chosen to expand the tensor channels from 1 to 8 with the first layer, then
from 8 to 16 with the second layer, using a kernel size of 3 on all dimensions. We also use the
adaptive average pooling module to reduce the dimensionality of the images to an 8 by 8 matrix,
which we will flatten in the forward pass to have a 1024 (16 * 8 * 8) resulting tensor.

Now let's see how the forward pass is defined.

```rust , ignore
# use burn::{
#     nn::{
#         conv::{Conv2d, Conv2dConfig},
#         pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
#         Dropout, DropoutConfig, Linear, LinearConfig, Relu,
#     },
#     prelude::*,
# };
#
# #[derive(Module, Debug)]
# pub struct Model<B: Backend> {
#     conv1: Conv2d<B>,
#     conv2: Conv2d<B>,
#     pool: AdaptiveAvgPool2d,
#     dropout: Dropout,
#     linear1: Linear<B>,
#     linear2: Linear<B>,
#     activation: Relu,
# }
#
# #[derive(Config, Debug)]
# pub struct ModelConfig {
#     num_classes: usize,
#     hidden_size: usize,
#     #[config(default = "0.5")]
#     dropout: f64,
# }
#
# impl ModelConfig {
#     /// Returns the initialized model.
#     pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
#         Model {
#             conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
#             conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
#             pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
#             activation: Relu::new(),
#             linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
#             linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
#             dropout: DropoutConfig::new(self.dropout).init(),
#         }
#     }
# }
#
impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // Create a channel at the second dimension.
        let x = images.reshape([batch_size, 1, height, width]);


        let x = self.conv1.forward(x); // [batch_size, 8, _, _]
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x); // [batch_size, 16, _, _]
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x) // [batch_size, num_classes]
    }
}
```

For former PyTorch users, this might feel very intuitive, as each module is directly incorporated
into the code using an eager API. Note that no abstraction is imposed for the forward method. You
are free to define multiple forward functions with the names of your liking. Most of the neural
network modules already built with Burn use the `forward` nomenclature, simply because it is
standard in the field.

Similar to neural network modules, the [`Tensor`](../building-blocks/tensor.md) struct given as a
parameter also takes the Backend trait as a generic argument, alongside its dimensionality. Even if
it is not used in this specific example, it is possible to add the kind of the tensor as a third
generic argument. For example, a 3-dimensional Tensor of different data types(float, int, bool)
would be defined as following:

```rust , ignore
Tensor<B, 3> // Float tensor (default)
Tensor<B, 3, Float> // Float tensor (explicit)
Tensor<B, 3, Int> // Int tensor
Tensor<B, 3, Bool> // Bool tensor
```

Note that the specific element type, such as `f16`, `f32` and the likes, will be defined later with
the backend.
