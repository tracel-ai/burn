# Custom CubeCL Kernel

This example fuses matrix multiplication, bias addition, and ReLU into one custom operation.
The [example project](https://github.com/tracel-ai/burn/tree/main/examples/custom-cubecl-kernel)
contains the source and dependencies. The code below is included directly from that project so
that changes to its API also update this chapter.

The CubeCL frontend compiles the kernel for the selected runtime. The example uses WGPU;
its `CubeBackend` implementation is shared by CubeCL runtimes.

## Custom Backend Trait

`#[backend_extension(Autodiff, Cube, Fusion)]` registers a trait with runtime dispatch. The low-level
signature uses `FloatTensor<Self>` primitives, while the public wrapper accepts `Tensor<3>`.
`into_dispatch()` and `from_dispatch()` cross that boundary without requiring a backend generic in
the caller's code. The reference implementation composes ordinary tensor operations for comparison.

```rust,ignore
{{#include ../../../../examples/custom-cubecl-kernel/src/lib.rs}}
```

The `Autodiff` selector enables routing to the handwritten autodiff implementation below. There is
no additional user-defined `AutodiffBackend` marker trait to implement.

## Lazy Fusion registration

Enable Burn's `fusion` feature and list `Fusion` on `#[backend_extension]` as shown above.
The annotation copies `lhs`'s dtype and passes borrowed `Shape` values to `output_shape`.
Fusion uses that metadata to register a lazy output, then calls the forward implementation
when the operation executes.

Sharing `output_shape` with execution keeps both paths consistent: it checks matrix dimensions,
batch broadcasting, and the requirement that bias match the output shape. The wrapper does not
compare output shapes with backend results, so the metadata calculation must be correct.

The custom kernel remains opaque to the Fusion optimizer. The wrapper does not combine it with
neighboring kernels or generate gradients; the handwritten `Autodiff<B, C>` implementation below
supplies the backward pass.

For structured outputs, use `#[fusion(meta = callable)]`. The callback must describe the same
result as direct backend execution, including the actual values of any non-tensor fields:
Fusion returns those values immediately and discards the backend's later values without comparison.
Use `#[fusion(default)]` to inherit an existing trait body, or omit `Fusion` from
`#[backend_extension]` to write the Fusion implementation yourself.

## Forward Kernel

The kernel computes each output from the corresponding row and column, then adds the bias and
applies ReLU. This is an instructional implementation; production matmul kernels use more advanced
tiling and hardware-specific optimizations.

```rust,ignore
{{#include ../../../../examples/custom-cubecl-kernel/src/kernel.rs}}
```

The forward implementation checks that operands share a device, makes them contiguous, allocates
the output, and launches the kernel with the required grid and element type.

```rust,ignore
{{#include ../../../../examples/custom-cubecl-kernel/src/forward.rs}}
```

## Backward

The custom trait is implemented for `Autodiff<B, C>` where the inner `B` implements the same trait.
The backward state saves the operands and shape information needed to compute the derivatives.
Tracked operations register that state with the graph; untracked operations only execute the
forward computation.

```rust,ignore
{{#include ../../../../examples/custom-cubecl-kernel/src/backward.rs}}
```

The backward pass masks the incoming gradient for ReLU, multiplies by transposed operands for the
matmul gradients, and reduces broadcast dimensions to match the original input and bias shapes.
Verify all three input gradients against the reference implementation, as well as forward values.

For application code, initialize inputs on an autodiff device and mark only source leaves whose
gradients you need. Backend generics stay inside these primitive implementations; callers use the
same `matmul_add_relu_custom` function for training and inference.
