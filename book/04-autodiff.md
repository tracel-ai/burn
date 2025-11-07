# Chapter 4: Automatic Differentiation

Automatic differentiation (autodiff) is the engine of modern deep learning, allowing frameworks to automatically compute the gradients (derivatives) of a model's parameters with respect to a loss function. Burn's implementation of autodiff is a prime example of its elegant, composable design, using a "backend decorator" pattern.

## The `Autodiff` Backend Decorator

The core of Burn's autodiff system is the `Autodiff<B>` struct, found in `crates/burn-autodiff/src/backend.rs`. It isn't a standalone backend; instead, it's a **decorator** that wraps another backend `B` to add autodiff capabilities to it.

Let's analyze its definition:

```rust
// crates/burn-autodiff/src/backend.rs

#[derive(Clone, Copy, Debug, Default)]
pub struct Autodiff<B, C = NoCheckpointing> {
    _b: PhantomData<B>,
    _checkpoint_strategy: PhantomData<C>,
}

impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    type Device = B::Device;

    type FloatTensorPrimitive = AutodiffTensor<B>;
    type FloatElem = B::FloatElem;

    type IntTensorPrimitive = B::IntTensorPrimitive;
    type IntElem = B::IntElem;

    // ... and so on for Bool and Quantized tensors
}
```

### Line-by-Line Analysis:

*   **`pub struct Autodiff<B, C = NoCheckpointing>`**:
    *   **`B`**: This generic parameter is the **inner backend** that `Autodiff` will wrap. This could be `Wgpu`, `NdArray`, or any other type that implements the `Backend` trait.
    *   **`C = NoCheckpointing`**: This parameter controls the checkpointing strategy for gradient computation, which is a memory-saving technique. It defaults to no checkpointing.
    *   The struct itself is empty, containing only `PhantomData`. This is because `Autodiff` is a zero-cost abstraction; it has no runtime data. Its purpose is to change the *type* of the backend, which in turn changes how the compiler resolves trait implementations.

*   **`impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C>`**:
    *   This is the key: `Autodiff<B>` **also implements the `Backend` trait**. This means it can be used anywhere a regular backend can. It's a "drop-in" replacement that adds new functionality.

*   **`type FloatTensorPrimitive = AutodiffTensor<B>;`**:
    *   Here's where the magic happens. For an `Autodiff` backend, the primitive type for float tensors is not the inner backend's primitive, but a new struct called `AutodiffTensor<B>`. This specialized tensor struct is what tracks the computation graph needed for backpropagation.

*   **`type IntTensorPrimitive = B::IntTensorPrimitive;`**:
    *   For integer and boolean tensors, the primitive type is simply passed through from the inner backend `B`. This is because you typically don't need to compute gradients for integer or boolean operations, so there's no need to track them in the computation graph.

### The Computation Graph

When you use an `Autodiff` backend, every operation on a tensor that requires a gradient is recorded as a node in a **computation graph**. This graph represents the mathematical relationships between the tensors.

Here is a diagram illustrating the graph created by the code example below:

```
+-----------+      +-----------+
| Tensor `x`|      | Tensor `y`|  (Leaf nodes, require_grad=true)
| [2.0, 5.0]|      | [7.0, 1.0]|
+-----------+      +-----------+
      |                  |
      `-------. .--------'
              | |
              V V
        +-----------+
        | Operation |      (Intermediate node)
        |    `*`    |
        +-----------+
              |
              V
        +-----------+
        | Tensor `z`|
        | [14.0, 5.0]|
        +-----------+
              |
              V
        +-----------+
        | Operation |
        |   `sum`   |
        +-----------+
              |
              V
       +--------------+
       | `final_tensor` |    (Root of the graph)
       |     19.0     |
       +--------------+
```
When you call `.backward()` on `final_tensor`, Burn traverses this graph backward from the root, applying the chain rule at each operation node to compute the gradient of `final_tensor` with respect to each leaf node (`x` and `y`).

### Code Example: A Manual Backward Pass

Here's how you would use the `Autodiff` backend to perform a simple calculation and compute gradients.

```rust
use burn::prelude::*;
use burn::backend::{Autodiff, NdArray};

// Define the backend. We'll use NdArray for CPU, wrapped with Autodiff.
type MyBackend = Autodiff<NdArray<f32>>;
type MyTensor<const D: usize> = Tensor<MyBackend, D>;

fn main() {
    let device = Default::default();

    // Create two tensors that require gradients.
    let x = MyTensor::<1>::from_data([2.0, 5.0], &device).require_grad();
    let y = MyTensor::<1>::from_data([7.0, 1.0], &device).require_grad();

    // Perform a calculation. Burn tracks these operations in a graph.
    // z = x * y
    let z = x.clone() * y.clone();
    // final = sum(z)
    let final_tensor = z.sum();

    // Compute the gradients by calling .backward() on the final tensor.
    let grads = final_tensor.backward();

    // Retrieve the gradients for x and y.
    let x_grad = x.grad(&grads).unwrap();
    let y_grad = y.grad(&grads).unwrap();

    println!("Gradient of x: {:?}", x_grad.to_data());
    // The gradient of `sum(x*y)` w.r.t. `x` is `y`. So, the output is [7.0, 1.0].

    println!("Gradient of y: {:?}", y_grad.to_data());
    // The gradient of `sum(x*y)` w.r.t. `y` is `x`. So, the output is [2.0, 5.0].
}
```

This design is incredibly powerful because it completely decouples the logic of automatic differentiation from the logic of the underlying tensor operations. You can create a new backend (e.g., for a new hardware accelerator) and get autodiff "for free" just by wrapping it with `Autodiff`.

---

## Exercises

1.  **Explore the Graph**: In the `burn-autodiff` crate, look at the `graph` directory. What role do you think the `Graph` and `Node` structs play in the process of automatic differentiation?
2.  **Calculate Gradients Manually**: Consider the function `f(x, y) = (x + y) * 2`.
    a.  Manually calculate the partial derivative of `f` with respect to `x` and the partial derivative of `f` with respect to `y`.
    b.  Write a Burn program similar to the example above to compute these gradients automatically for `x = 3.0` and `y = 4.0`. Do your manual calculations match Burn's output?
3.  **No Grad**: What happens if you remove the `.require_grad()` call from the `x` tensor in the main code example? What is the value of `x.grad(&grads)`? Why is this behavior useful?
4.  **Chain Rule**: Implement the function `z = sin(x * y)` in Burn. Calculate the gradient of `z` with respect to `x` and `y` at `x=2.0, y=3.0`. Manually verify the result using the chain rule. (Hint: `dz/dx = cos(x*y)*y`).
