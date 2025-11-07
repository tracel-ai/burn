# Chapter 2: The Heart of Burn - Tensors

At the absolute core of any deep learning framework is the `Tensor`. This chapter provides a deep, line-by-line analysis of Burn's `Tensor` implementation, found in the `burn-tensor` crate. Understanding this central data structure is the key to mastering the entire framework.

## The Tensor Struct: A Generic Powerhouse

The primary definition of the `Tensor` struct is found in `crates/burn-tensor/src/tensor/api/base.rs`. Let's break it down:

```rust
// crates/burn-tensor/src/tensor/api/base.rs

#[derive(new, Clone, Debug)]
pub struct Tensor<B, const D: usize, K = Float>
where
    B: Backend,
    K: TensorKind<B>,
{
    pub(crate) primitive: K::Primitive,
}
```

### Line-by-Line Analysis:

*   **`pub struct Tensor<B, const D: usize, K = Float>`**: This is the declaration.
    *   **`B: Backend`**: This is the first and most important generic parameter. It represents the **Backend**, which is the engine that will perform the actual computations (e.g., a CPU backend like `ndarray` or a GPU backend like `wgpu`). By making the backend a generic parameter, Burn allows you to write your code once and run it on different hardware.
    *   **`const D: usize`**: This is a *const generic*, representing the **dimensionality** or **rank** of the tensor at compile time. A 2D tensor (a matrix) would have `D = 2`, a 3D tensor would have `D = 3`, and so on. This allows Rust's type system to catch shape-related errors at compile time, which is a major advantage over frameworks like PyTorch where such errors only appear at runtime.
    *   **`K = Float`**: This generic parameter represents the **Kind** of the tensor. It defaults to `Float`, meaning it's a tensor of floating-point numbers. Other kinds include `Int` for integers and `Bool` for booleans. This ensures type safety for tensor operations (e.g., you can't perform a matrix multiplication on boolean tensors).

*   **`where B: Backend, K: TensorKind<B>`**: These are trait bounds.
    *   `B: Backend` ensures that the type `B` implements the `Backend` trait, which defines all the necessary tensor operations.
    *   `K: TensorKind<B>` links the tensor kind (`Float`, `Int`, etc.) to the backend. This allows each backend to have its own specific implementation for different kinds of tensors.

*   **`pub(crate) primitive: K::Primitive`**: This is the single field of the `Tensor` struct.
    *   It holds the `primitive`, which is the actual, backend-specific tensor object. `K::Primitive` is an associated type on the `TensorKind` trait. For example, for a `wgpu` backend and a `Float` tensor, this might be a struct that holds a handle to a GPU buffer.
    *   This design is a classic example of the **handle pattern**. The `burn_tensor::Tensor` is a lightweight, generic handle that delegates all the real work to the backend-specific `primitive` it contains.

### Ownership Model

The ownership model of the `Tensor` is simple and efficient.

```
Tensor<B, D, K> (on the stack)
└── primitive: K::Primitive (owned)
    └── (Backend-specific data, e.g., a handle to a GPU buffer or a heap-allocated array)
```

*   A `Tensor` struct itself is very small and lives on the stack.
*   It **owns** its `primitive`. When a `Tensor` is dropped, its `primitive` is also dropped, which in turn is responsible for cleaning up its resources (e.g., freeing the GPU buffer).
*   The `Tensor` struct implements `Clone`. When you clone a `Tensor`, you are typically performing a cheap, reference-counted clone of the underlying data, not a deep copy of the entire tensor's contents. This is a common and important optimization.

## Interacting with the Tensor API

The `Tensor` struct provides a rich API for creating and manipulating tensors. All of these API calls are high-level and backend-agnostic; they delegate the actual work to the underlying backend.

### Creation

```rust
use burn::prelude::*;

fn creation<B: Backend>(device: &B::Device) {
    // Create a 2x3 tensor of zeros
    let zeros = Tensor::<B, 2>::zeros([2, 3], device);

    // Create a 2x3 tensor of ones
    let ones = Tensor::<B, 2>::ones([2, 3], device);

    // Create a tensor from existing data
    let data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let tensor_from_data = Tensor::<B, 2>::from_data(data, device);
}
```

### Manipulation

The API provides a comprehensive set of methods for changing a tensor's shape and layout.

```rust
use burn::prelude::*;

fn manipulation<B: Backend>(device: &B::Device) {
    let tensor = Tensor::<B, 3>::ones([2, 3, 4], device);

    // Reshape the tensor to a new shape
    // The total number of elements must remain the same.
    let reshaped = tensor.clone().reshape([2, 12]);
    // reshaped.dims() will be [2, 12]

    // Get a slice of the tensor
    // This selects rows 0-1 and columns 1-2
    let slice = tensor.clone().slice([0..2, 1..3]);
    // slice.dims() will be [2, 2, 4]

    // Perform matrix multiplication
    // The const generic `D` helps ensure shape compatibility at compile time.
    let tensor_2d = Tensor::<B, 2>::ones([4, 5], device);
    // Note: this won't compile because of shape mismatch, a safety feature!
    // let matmul_result = tensor.matmul(tensor_2d);
}
```

### Arithmetic

Element-wise arithmetic operations are supported through standard Rust operators.

```rust
use burn::prelude::*;

fn arithmetic<B: Backend>(device: &B::Device) {
    let t1 = Tensor::<B, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], device);
    let t2 = Tensor::<B, 2>::from_data([[5.0, 6.0], [7.0, 8.0]], device);

    // Element-wise addition
    let sum = t1.clone() + t2.clone();
    // sum will be [[6.0, 8.0], [10.0, 12.0]]

    // Element-wise multiplication
    let product = t1 * t2;
    // product will be [[5.0, 12.0], [21.0, 32.0]]
}
```
This function perfectly illustrates Burn's design: the high-level API performs checks and handles data conversion, then delegates the low-level, hardware-specific work to the backend through a trait. In the next chapter, we will see in more detail the `Backend` system.

---

## Exercises

1.  **Tensor Creation**: Create a 3D tensor of integers with the shape `[2, 2, 3]` filled with random values. Use the `Tensor::random` method and a `Distribution`. (Hint: You'll need to specify the `Int` kind for the tensor).
2.  **Reshaping and Slicing**:
    a.  Take the tensor from exercise 1 and reshape it into a 2D tensor of shape `[4, 3]`.
    b.  From the reshaped tensor, extract a slice containing the second and third rows.
3.  **Arithmetic and Broadcasting**:
    a.  Create a 2D tensor `A` of shape `[3, 4]` filled with the value `5.0`.
    b.  Create a 1D tensor `B` of shape `[4]` filled with the value `2.0`.
    c.  Compute `A + B`. What is the shape of the result? This is an example of **broadcasting**. Read about broadcasting in the context of tensor libraries and explain why this operation is valid.
