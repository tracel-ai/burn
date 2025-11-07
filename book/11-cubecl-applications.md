# Chapter 11: CubeCL Applications

The previous chapter introduced CubeCL as a powerful "meta-backend" for writing portable, high-performance GPU kernels. This chapter will showcase a practical application of CubeCL by analyzing the "custom-cubecl-kernel" example from the Burn repository. This example demonstrates one of the key benefits of CubeCL: **kernel fusion**.

## The Goal: Fusing Operations

In a typical neural network, a linear layer is often followed by an activation function. The computation looks like this:

1.  **Matrix Multiplication**: `output = input.matmul(weights)`
2.  **Bias Addition**: `output = output + bias`
3.  **Activation**: `output = relu(output)`

On a GPU, this would normally require launching three separate kernels. Each launch has overhead, which can add up. The goal of this example is to fuse these three distinct operations into a **single GPU kernel**, which will be more efficient.

## The Fused Kernel: `fused_matmul_add_relu`

The custom kernel is defined in `examples/custom-cubecl-kernel/src/kernel.rs`.

```rust
// examples/custom-cubecl-kernel/src/kernel.rs

use cubecl::{cube, prelude::*};

/// Declare a custom kernel that gets compiled to `wgpu`/`CUDA`
#[cube(launch)]
pub fn fused_matmul_add_relu_kernel<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    bias: &Tensor<F>,
    output: &mut Tensor<F>,
) {
    // ... kernel logic ...

    let mut sum = F::new(0.0);
    for k in 0..dim_k {
        // ... matrix multiplication logic ...
        sum += lhs[lhs_index] * rhs[rhs_index];
    }

    // ...

    // Bias addition and ReLU activation are fused here
    output[index] = F::max(sum + bias[index], F::new(0.0));
}
```

### Line-by-Line Analysis:

*   **`#[cube(launch)]`**: This is the CubeCL macro that marks this function as a GPU kernel. The `launch` argument indicates that this is a "launch kernel," the main entry point for a GPU computation.
*   **`pub fn fused_matmul_add_relu_kernel<F: Float>(...)`**:
    *   The function is generic over `F: Float`, meaning it can work with different floating-point types (like `f32`, `f16`, etc.).
    *   It takes the left-hand side (`lhs`) and right-hand side (`rhs`) tensors for the matrix multiplication, a `bias` tensor, and a mutable `output` tensor to write the results to.
*   **Kernel Logic**:
    *   The first part of the kernel calculates the correct indices for each GPU thread to work on its part of the matrix multiplication.
    *   The `for` loop performs the core dot-product computation for a single element of the output matrix and stores the result in `sum`.
*   **The Fusion**:
    *   **`output[index] = F::max(sum + bias[index], F::new(0.0));`**: This single line is where the magic happens.
        *   `sum + bias[index]`: The bias is added to the result of the matrix multiplication.
        *   `F::max(..., F::new(0.0))`: The result is then passed through a `max` function with 0.0, which is the mathematical equivalent of the ReLU activation function (`max(0, x)`).

This single kernel performs all three operations without the need for intermediate tensors or separate kernel launches.

## Integrating the Kernel into Burn

Once the kernel is defined, it needs to be integrated into Burn's operation system. This involves creating forward and backward passes.

*   **Forward Pass (`forward.rs`)**: A new `_fused_matmul_add_relu` function is defined. This function is the high-level entry point that Burn's `Tensor` API will call. Inside this function, a `JitAutotuneOperationSet` is used. This is a powerful feature of CubeCL that can benchmark different versions of a kernel (e.g., with different tile sizes or launch parameters) at runtime and automatically select the fastest one for the current hardware. The selected kernel is then launched using `client.execute`.

*   **Backward Pass (`backward.rs`)**: To make this operation trainable within Burn's `Autodiff` system, a custom backward pass must be defined. This is a more advanced topic that requires a good understanding of the chain rule and matrix calculus. The backward pass for this fused operation involves defining separate kernels for computing the gradients with respect to the `lhs`, `rhs`, and `bias` inputs. These backward kernels are then registered with the autodiff graph.

## Why This Matters

This example perfectly illustrates the power of CubeCL:

*   **Performance**: Fusing these operations reduces kernel launch overhead and memory traffic (no need to write intermediate results back to global memory), resulting in a significant speedup.
*   **Abstraction**: The kernel is written in a high-level, Rust-like language. The same kernel code can be compiled by CubeCL to run on different GPU backends (like `wgpu` and `cuda`) without any changes.
*   **Extensibility**: If you have a specific, performance-critical operation in your model that isn't a standard primitive in Burn, you can use CubeCL to write your own custom, high-performance kernel for it.

CubeCL provides a path for deep optimization and customization, allowing you to get the most out of your hardware while maintaining the portability and safety of the Burn ecosystem.

---

## Exercises

1.  **Modify the Kernel**:
    a.  Create a copy of the `fused_matmul_add_relu_kernel`.
    b.  Modify your new kernel to implement a "Leaky ReLU" activation instead of a standard ReLU. The formula for Leaky ReLU is `max(alpha * x, x)`, where `alpha` is a small constant (e.g., 0.01).
    c.  You will need to pass `alpha` as a new argument to your kernel.
2.  **Explore the Backward Pass**: Look at the `backward.rs` file in the example. You will see kernels like `lhs_grad` and `rhs_grad`. At a high level, what mathematical operations do these backward kernels correspond to in the context of matrix multiplication? (Hint: Think about the derivatives of `C = A * B`).
3.  **Thought Experiment**: Kernel fusion is a powerful optimization. What are some other sequences of common neural network operations that would be good candidates for fusion?
