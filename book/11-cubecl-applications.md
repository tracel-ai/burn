# Chapter 11: CubeCL Applications

The previous chapter introduced CubeCL as a powerful "meta-backend" for writing portable, high-performance GPU kernels. This chapter will showcase a practical application of CubeCL by analyzing the "custom-cubecl-kernel" example from the Burn repository. This example demonstrates one of the key benefits of CubeCL: **kernel fusion**.

## The Goal: Fusing Operations

In a typical neural network, a linear layer is often followed by an activation function. The computation looks like this:

1.  **Matrix Multiplication**: `output = input.matmul(weights)`
2.  **Bias Addition**: `output = output + bias`
3.  **Activation**: `output = relu(output)`

On a GPU, this would normally require launching three separate kernels. Each launch has overhead, which can add up. The goal of this example is to fuse these three distinct operations into a **single GPU kernel**, which will be more efficient.

### Kernel Fusion Diagram
```
Without Fusion:
+------------+   +----------+   +--------+
| Matmul     |   | Add Bias |   | ReLU   |
| Kernel     |-->| Kernel   |-->| Kernel |
+------------+   +----------+   +--------+
(3 separate GPU kernel launches, 2 intermediate memory writes/reads)

With Fusion:
+--------------------------+
| Fused MatmulAddRelu      |
| Kernel                   |
+--------------------------+
(1 GPU kernel launch, 0 intermediate memory writes/reads)
```
As the diagram shows, fusion reduces the number of kernel launches and, just as importantly, avoids writing the intermediate results of the `matmul` and `add` operations to global GPU memory and then reading them back. This reduction in memory traffic is often a significant source of performance improvement.

## The Fused Kernel: `fused_matmul_add_relu`

The custom kernel is defined in `examples/custom-cubecl-kernel/src/kernel.rs`.

```rust
// examples/custom-cubecl-kernel/src/kernel.rs

use cubecl::{cube, prelude::*};

#[cube(launch)]
pub fn fused_matmul_add_relu_kernel<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    bias: &Tensor<F>,
    output: &mut Tensor<F>,
) {
    // ... index calculation logic ...

    let mut sum = F::new(0.0);
    for k in 0..dim_k {
        // ... matrix multiplication logic ...
        sum += lhs[lhs_index] * rhs[rhs_index];
    }

    // Bias addition and ReLU activation are fused here
    output[index] = F::max(sum + bias[index], F::new(0.0));
}
```

## Integrating the Kernel into Burn

Once the kernel is defined, it needs to be integrated into Burn's operation system.

### Forward Pass (`forward.rs`)

A new `_fused_matmul_add_relu` function is defined. This is the high-level entry point that Burn's `Tensor` API will call.

```rust
// examples/custom-cubecl-kernel/src/forward.rs

// ... (imports and trait definitions)

/// Fused matmul, add, relu operation
pub fn _fused_matmul_add_relu<R: CubeRuntime, F: FloatElement, I: IntElement, const D: usize>(
    lhs: CubeTensor<R, F, D>,
    rhs: CubeTensor<R, F, D>,
    bias: CubeTensor<R, F, D>,
) -> CubeTensor<R, F, D> {
    // ... (shape calculations and output tensor creation)

    let kernel = JitAutotuneOperationSet::new(
        vec![
            // Multiple kernel implementations can be provided here for autotuning.
            // CubeCL will benchmark them and pick the fastest one.
            Box::new(JitOperation::new(FusedMatmulAddReluAutotuneKey::new(
                "fused_matmul_add_relu".to_string(),
            ))),
        ],
    );

    let stream = lhs.client.stream();
    stream.execute(Box::new(kernel), &[&lhs, &rhs, &bias, &output]);

    output
}
```
This function uses a `JitAutotuneOperationSet`. This is a powerful feature of CubeCL that can benchmark different versions of a kernel at runtime and automatically select the fastest one for the current hardware. The selected kernel is then launched using `stream.execute`.

### Backward Pass (`backward.rs`)

To make this operation trainable, a custom backward pass is defined. This involves defining separate kernels for computing the gradients with respect to the `lhs`, `rhs`, and `bias` inputs.

```rust
// examples/custom-cubecl-kernel/src/backward.rs

// Defines the backward step for our fused operation
impl<B: AutodiffBackend> FusedMatmulAddRelu<B>
where
    // ... (trait bounds)
{
    fn backward(self, grads: &mut Gradients) {
        // ... (kernels for lhs_grad, rhs_grad, and bias_grad are defined here)

        // Register the backward operations in the autodiff graph
        let mut lhs_grad_node = UnaryGrad::new(self.output, self.lhs, grads);
        let mut rhs_grad_node = UnaryGrad::new(self.output.clone(), self.rhs, grads);
        let mut bias_grad_node = UnaryGrad::new(self.output.clone(), self.bias, grads);

        lhs_grad_node.register_closure(move |out_grad| _lhs_grad(out_grad, ...));
        rhs_grad_node.register_closure(move |out_grad| _rhs_grad(out_grad, ...));
        bias_grad_node.register_closure(move |out_grad| _bias_grad(out_grad, ...));
    }
}
```
This is a simplified view, but it shows the core concept: for each input to the forward pass that requires a gradient, you must provide a function that computes that gradient. These functions are then registered with the autodiff graph.

## Why This Matters

This example perfectly illustrates the power of CubeCL:

*   **Performance**: Fusing these operations reduces kernel launch overhead and memory traffic, resulting in a significant speedup.
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
