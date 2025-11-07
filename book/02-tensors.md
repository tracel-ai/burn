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

### Creation and Manipulation Examples

```rust
use burn::prelude::*;

fn tensor_operations<B: Backend>(device: &B::Device) {
    // === Creation ===
    let tensor = Tensor::<B, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], device);

    // === Combining Tensors ===
    // `cat` concatenates along an existing dimension.
    let t1 = Tensor::<B, 2>::ones([2, 3], device);
    let t2 = Tensor::<B, 2>::zeros([2, 2], device);
    let t_cat = Tensor::cat(vec![t1, t2], 1); // Concatenate along dimension 1 (columns)
    // t_cat shape will be [2, 5]

    // `stack` creates a new dimension.
    let t3 = Tensor::<B, 2>::ones([2, 3], device);
    let t4 = Tensor::<B, 2>::zeros([2, 3], device);
    let t_stack = Tensor::stack(vec![t3, t4], 0); // Stack along a new dimension 0
    // t_stack shape will be [2, 2, 3]

    // === Shape Manipulation ===
    let original = Tensor::<B, 2>::ones([4, 6], device);

    // `reshape` changes the dimensions while keeping the data layout.
    let reshaped = original.clone().reshape([2, 12]);
    // reshaped shape is [2, 12]

    // `transpose` swaps the last two dimensions.
    let transposed = original.clone().transpose();
    // transposed shape is [6, 4]

    // `permute` reorders the dimensions to a new order.
    let permuted = original.clone().permute([1, 0]);
    // permuted shape is [6, 4]

    // === Reduction Operations ===
    let tensor_to_reduce = Tensor::<B, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device);

    // `sum` reduces the entire tensor to a single scalar tensor.
    let total_sum = tensor_to_reduce.clone().sum();
    // total_sum will contain the value 21.0

    // `sum_dim` reduces along a specific dimension.
    let sum_cols = tensor_to_reduce.clone().sum_dim(1);
    // sum_cols will be a tensor with data [6.0, 15.0] and shape [2, 1]

    // `mean` and `mean_dim` work similarly for the average.
    let total_mean = tensor_to_reduce.clone().mean();
    // total_mean will contain the value 3.5
}
```

### Visualizing `reshape` vs. `transpose`

It's important to understand the difference between `reshape` and operations like `transpose`. `reshape` changes the dimensions but does not change the order of the data in memory. `transpose` (and `permute`) changes how the dimensions are interpreted, effectively re-ordering the data.

```
Original Tensor (Shape [2, 3], Data in memory: [1, 2, 3, 4, 5, 6])
[[1, 2, 3],
 [4, 5, 6]]

Reshape to [3, 2]:
The data layout is unchanged.
[[1, 2],
 [3, 4],
 [5, 6]]

Transpose:
The data layout is re-ordered conceptually.
[[1, 4],
 [2, 5],
 [3, 6]]
```

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
4.  **`cat` vs `stack`**:
    a. Create two 2D tensors of shape `[3, 5]`.
    b. Concatenate them along dimension 0. What is the resulting shape?
    c. Stack them along dimension 0. What is the resulting shape?
    d. Explain the difference in the output shapes.
