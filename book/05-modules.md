# Chapter 5: Building Neural Network Modules

So far, we've explored the core data structures (`Tensor`) and the computation engines (`Backend`). Now, we'll move up the stack to `burn-nn`, the crate that provides the building blocks for creating neural networks. The central concept in `burn-nn` is the `Module` trait.

## The `Module` Trait: Composable Building Blocks

The `Module` trait, defined in `crates/burn-core/src/module/base.rs`, is the foundation for all neural network components in Burn. Any struct that represents a part of a neural network—from a single layer to a complete model—will implement this trait.

The real power of the `Module` trait comes from the `#[derive(Module)]` procedural macro. When you add this attribute to your struct, the compiler automatically generates a significant amount of boilerplate code for you, making your module:

*   **Trainable**: The macro automatically discovers all fields of type `Param<T>` (learnable parameters) within your struct, making them accessible to optimizers.
*   **Serializable**: It generates implementations for `into_record()` and `load_record()`, allowing you to easily save and load your model's state (its learned parameters).
*   **Visitable**: It provides `visit()` and `map()` methods, which allow you to traverse the entire module tree and inspect or modify every parameter. This is useful for tasks like custom weight initialization or applying weight decay.

## A Concrete Example: The `Linear` Layer

Let's analyze the `Linear` layer, a fundamental component of most neural networks, found in `crates/burn-nn/src/modules/linear.rs`.

```rust
// crates/burn-nn/src/modules/linear.rs

#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Linear<B: Backend> {
    pub weight: Param<Tensor<B, 2>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
}
```

### Line-by-Line Analysis:

*   **`#[derive(Module, Debug)]`**: This is the key. The `Module` derive macro inspects the struct and generates the necessary implementations. `Debug` is also derived for easy printing.
*   **`pub struct Linear<B: Backend>`**: Like everything else in Burn, the `Linear` layer is generic over a `Backend`. This means the same `Linear` struct can be used on a CPU, a GPU, or any other backend.
*   **`pub weight: Param<Tensor<B, 2>>`**: This defines the `weight` parameter of the linear layer.
    *   The `Tensor<B, 2>` is a 2D tensor (a matrix) that will hold the weights.
    *   This tensor is wrapped in `Param<...>`. The `Param` struct is a special container that marks its contents as a **learnable parameter** of the module. The `#[derive(Module)]` macro looks for fields of this type.
*   **`pub bias: Option<Param<Tensor<B, 1>>>`**: This defines the optional `bias` parameter, which is a 1D tensor (a vector). It's an `Option` because a linear layer doesn't always need a bias term.

### The Forward Pass

A module is not complete without its `forward` method, which defines the computation to be performed.

```rust
// crates/burn-nn/src/modules/linear.rs

impl<B: Backend> Linear<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        linear(
            input,
            self.weight.val(),
            self.bias.as_ref().map(|b| b.val()),
        )
    }
}
```

*   **`pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D>`**:
    *   The `forward` function takes an `input` tensor and returns an `output` tensor.
    *   It's a convention in Burn to name this function `forward`, but it's not strictly required by the `Module` trait itself.
*   **`self.weight.val()`**: To access the underlying tensor from a `Param`, you call the `.val()` method.
*   **`linear(...)`**: This function, from `burn::tensor::module::linear`, performs the actual linear transformation (`O = IW + b`). It calls the appropriate backend functions to perform the matrix multiplication and bias addition.

### Ownership and Composition

Modules are designed to be composed. You can build a larger module by including other modules as fields.

```rust
use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, ReLU};

#[derive(Module, Debug)]
pub struct MyModel<B: Backend> {
    linear1: Linear<B>,
    relu: ReLU,
    linear2: Linear<B>,
}

#[derive(Config)]
pub struct MyModelConfig {
    d_input: usize,
    d_hidden: usize,
    d_output: usize,
}

impl MyModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MyModel<B> {
        MyModel {
            linear1: LinearConfig::new(self.d_input, self.d_hidden).init(device),
            relu: ReLU::new(),
            linear2: LinearConfig::new(self.d_hidden, self.d_output).init(device),
        }
    }
}

impl<B: Backend> MyModel<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = self.relu.forward(x);
        let x = self.linear2.forward(x);
        x
    }
}
```

Here's an ASCII diagram of the ownership:

```
`MyModel<B>` (owns its fields)
├── linear1: `Linear<B>`
│   ├── weight: `Param<Tensor<B, 2>>`
│   └── bias: `Option<Param<Tensor<B, 1>>>`
├── relu: `ReLU` (no parameters)
└── linear2: `Linear<B>`
    ├── weight: `Param<Tensor<B, 2>>`
    └── bias: `Option<Param<Tensor<B, 1>>>`
```

When you call `#[derive(Module)]` on `MyModel`, the macro will **recursively** find all the parameters in `linear1` and `linear2`. This means that when you save or load `MyModel`, you are saving or loading the state of the entire network. This composability is what allows you to build complex models from simple, reusable blocks.

---

## Exercises

1.  **Create a Custom Module**:
    a.  Create a new module called `GELU` that implements the GELU activation function. You can find the formula for GELU online. (Hint: The `erf` function is available in `burn::tensor::special`).
    b.  Derive `Module` and `Debug` for your `GELU` struct.
    c.  Implement a `forward` method for it.
2.  **Compose Modules**:
    a.  Modify the `MyModel` example to replace the `ReLU` activation with your new `GELU` module.
    b.  Instantiate your new model and perform a forward pass with a random tensor.
3.  **Count Parameters**: Use the `.num_params()` method on an instance of `MyModel`. How many total parameters does it have? Verify this by manually calculating the number of parameters in `linear1` and `linear2` (weight + bias).
