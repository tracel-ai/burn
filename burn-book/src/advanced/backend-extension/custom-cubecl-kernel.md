# Custom `cubecl` Kernel

In this section, you will learn how to create your own custom operation by writing your own kernel
with the cubecl compiler frontend. We will take the example of a common workflow in the deep
learning field, where we create a kernel to fuse multiple operations together. Note that `burn` does
this automatically, but a manual implementation might be more efficient in some cases. We will fuse
a matmul kernel followed by an addition and the ReLU activation function, which is commonly found in
various models. All the code can be found under the
[examples directory](https://github.com/tracel-ai/burn/tree/main/examples/custom-cubecl-kernel).

## Custom Backend Trait

First, we need to determine the type signature of our newly created operation by defining our custom
backend traits. As we will use the associated type `TensorPrimitive` of the `Backend` trait, which
encapsulates the underlying tensor implementation of the backend, we will use a type alias to avoid
the ugly disambiguation with associated types.

```rust, ignore
/// We create our own Backend trait that extends the Burn backend trait.
pub trait Backend: burn::tensor::backend::Backend {
    fn fused_matmul_add_relu(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

/// We create our own AutodiffBackend trait that extends the Burn autodiff backend trait.
pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}
```

In our project, we can use these traits instead of the
`burn::tensor::backend::{Backend, AutodiffBackend}` traits provided by Burn. Burn's user APIs
typically make use of the `Tensor` struct rather than dealing directly with primitive tensor types.
Therefore, we can encapsulate our newly defined backend traits with functions that expose new
operations while maintaining a consistent API.

```rust, ignore
/// We define our custom implementation using the added function on our custom backend.
pub fn matmul_add_relu_custom<B: Backend>(
    lhs: Tensor<B, 3>,
    rhs: Tensor<B, 3>,
    bias: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let output = B::fused_matmul_add_relu(
        lhs.into_primitive().tensor(),
        rhs.into_primitive().tensor(),
        bias.into_primitive().tensor(),
    );

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// We define a reference implementation using basic tensor operations.
pub fn matmul_add_relu_reference<B: Backend>(
    lhs: Tensor<B, 3>,
    rhs: Tensor<B, 3>,
    bias: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let x = lhs.matmul(rhs) + bias;

    activation::relu(x)
}

```

Note that we also provide a reference implementation for testing purposes, which allows us to easily
validate our new implementation. While not mandatory, having a reference implementation can be
valuable, especially in projects where creating a reference implementation solely using basic tensor
operations is feasible.

## Forward Kernel

Now, let's proceed to write the fused kernel using the `cubecl` compiler frontend. To keep things
simple, we'll create a straightforward matmul kernel without employing any intricate techniques. We
won't delve into the details of the `cube` macro, but if you're interested to learn more, please see
[`cubecl` Book](https://github.com/tracel-ai/cubecl/tree/f5b63076a01a5c03ea9ed20799d3eeaf776b45da/cubecl-book).
the The actual matmul, add and relu computations are found at the end, after an extensive prelude
that serves to correctly map each compute unit to the data it is responsible for, with support for
batches.

```rust, ignore
use cubecl::{cube, prelude::*};

#[cube(launch)]
pub fn fused_matmul_add_relu_kernel<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    bias: &Tensor<F>,
    output: &mut Tensor<F>,
) {
    let row = ABSOLUTE_POS_X;
    let col = ABSOLUTE_POS_Y;
    let batch = ABSOLUTE_POS_Z;

    let n_rows = output.shape(output.rank() - 2);
    let n_cols = output.shape(output.rank() - 1);
    let dim_k = rhs.shape(rhs.rank() - 1);

    if row >= n_rows || col >= n_cols {
        return;
    }

    let offset_output = batch * n_rows * n_cols;
    let mut offset_lhs = 0;
    let mut offset_rhs = 0;

    let batch_dims = output.rank() - 2;
    for dim in 0..batch_dims {
        offset_lhs += offset_output / output.stride(dim) % lhs.shape(dim) * lhs.stride(dim);
        offset_rhs += offset_output / output.stride(dim) % rhs.shape(dim) * rhs.stride(dim);
    }

    let mut sum = F::new(0.0);
    for k in 0..dim_k {
        let lhs_index = row * dim_k + k;
        let rhs_index = k * n_cols + col;

        sum += lhs[offset_lhs + lhs_index] * rhs[offset_rhs + rhs_index];
    }

    let out_index = row * n_cols + col;
    let index = offset_output + out_index;

    output[index] = F::max(sum + bias[index], F::new(0.0));
}
```

Now, let's move on to the next step, which involves implementing the remaining code to launch the
kernel. We'll go into implementing our custom backend trait for the generic JIT backend. This
automatically implements the trait for `burn-cuda`, `burn-wgpu` as well as fusion.

```rust, ignore
/// Implement our custom backend trait for the generic `CubeBackend`.
impl<R: CubeRuntime, F: FloatElement, I: IntElement> Backend for CubeBackend<R, F, I> {
    fn fused_matmul_add_relu(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        // Define cube dim, hardcoded for simplicity.
        let cube_dim = CubeDim { x: 16, y: 16, z: 1 };

        lhs.assert_is_on_same_device(&rhs);
        lhs.assert_is_on_same_device(&bias);

        // For simplicity, make sure each tensor is continuous.
        let lhs = into_contiguous(lhs);
        let rhs = into_contiguous(rhs);
        let bias = into_contiguous(bias);

        // Get the matmul relevant shapes.
        let ndims = lhs.shape.num_dims();
        let num_rows = lhs.shape.dims[ndims - 2];
        let num_cols = rhs.shape.dims[ndims - 1];

        // Compute shape of output, while tracking number of batches.
        let mut num_batches = 1;
        let mut shape_out = vec![0; ndims];
        for i in shape_out.clone().into_iter().take(ndims - 2) {
            shape_out[i] = usize::max(lhs.shape.dims[i], rhs.shape.dims[i]);
            num_batches *= shape_out[i];
        }
        shape_out[ndims - 2] = num_rows;
        shape_out[ndims - 1] = num_cols;
        let shape_out = Shape::from(shape_out);

        // Create a buffer for the output tensor.
        let buffer = lhs
            .client
            .empty(shape_out.num_elements() * core::mem::size_of::<F>());

        // Create the output tensor primitive.
        // Create the output tensor primitive.
        let output = CubeTensor::new_contiguous(
            lhs.client.clone(),
            lhs.device.clone(),
            shape_out,
            buffer,
            F::dtype(),
        );

        // Declare the wgsl workgroup with the number of cubes in x, y and z.
        let cubes_needed_in_x = f32::ceil(num_rows as f32 / cube_dim.x as f32) as u32;
        let cubes_needed_in_y = f32::ceil(num_cols as f32 / cube_dim.y as f32) as u32;
        let cube_count =
            CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, num_batches as u32);

        // Execute lazily the kernel with the launch information and the given buffers. For
        // simplicity, no vectorization is performed
        fused_matmul_add_relu_kernel::launch::<F, R>(
            &lhs.client,
            cube_count,
            cube_dim,
            lhs.as_tensor_arg::<F>(1),
            rhs.as_tensor_arg::<F>(1),
            bias.as_tensor_arg::<F>(1),
            output.as_tensor_arg::<F>(1),
        );

        // Return the output tensor.
        output
    }
}
```

In the preceding code block, we demonstrated how to launch the kernel that modifies the correct
buffer. It's important to note that Rust's mutability safety doesn't apply here; the context has the
capability to execute any mutable operation on any buffer. While this isn't a problem in the
previous scenario where we only modify the newly created output buffer, it is wise to keep this in
mind.

## Backward

Now that the custom backend trait is implemented for the JIT backend, you can use it to invoke the
`matmul_add_relu_custom` function. However, calculating gradients is not yet possible at this stage.
If your use case does not extend beyond inference, there is no need to implement any of the
following code.

For the backward pass, we will leverage the backend implementation from `burn-autodiff`, which is
actually generic over the backend. Instead of crafting our own `cubecl` kernel for the backward
pass, we will use our fused kernel only for the forward pass, and compute the gradient using basic
operations.

```rust, ignore
// Implement our custom backend trait for any backend that also implements our custom backend trait.
impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    fn fused_matmul_add_relu(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        // Create our zero-sized type that will implement the Backward trait.
        #[derive(Debug)]
        struct FusedMatmulAddReluBackward;

        // Implement the backward trait for the given backend B, the node gradient
        // with three other gradients to calculate (lhs, rhs, and bias).
        impl<B: Backend> Backward<B, 3> for FusedMatmulAddReluBackward {
            // Our state that we must build during the forward pass to compute the backward pass.
            //
            // Note that we could improve the performance further by only keeping the state of
            // tensors that are tracked, improving memory management, but for simplicity, we avoid
            // that part.
            type State = (NodeID, NodeID, FloatTensor<B>, Shape);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                // Get the nodes of each variable.
                let [node_lhs, node_rhs, node_bias] = ops.parents;
                // Fetch the gradient for the current node.
                let grad = grads.consume::<B>(&ops.node);

                // Set our state.
                let (lhs_state, rhs_state, output, shape_bias) = ops.state;
                let lhs: FloatTensor<B> = checkpointer.retrieve_node_output(lhs_state);
                let rhs: FloatTensor<B> = checkpointer.retrieve_node_output(rhs_state);

                // Fetch shapes of our tensor to support broadcasting.
                let shape_lhs = lhs.shape();
                let shape_rhs = rhs.shape();

                // Compute the gradient of the output using the already existing `relu_backward`
                // function in the basic Burn backend trait.
                let grad_output = B::relu_backward(output, grad);

                // Compute the lhs gradient, which is the derivative of matmul with support for
                // broadcasting.
                let grad_lhs = broadcast_shape::<B>(
                    B::float_matmul(grad_output.clone(), B::float_transpose(rhs)),
                    &shape_lhs,
                );
                // Compute the rhs gradient, which is the derivative of matmul with support for
                // broadcasting.
                let grad_rhs = broadcast_shape::<B>(
                    B::float_matmul(B::float_transpose(lhs), grad_output.clone()),
                    &shape_rhs,
                );
                // The add derivative is only 1, so we just need to support broadcasting to
                // compute the bias gradient.
                let grad_bias = broadcast_shape::<B>(grad_output, &shape_bias);

                // Register the gradient for each variable based on whether they are marked as
                // `tracked`.
                if let Some(node) = node_bias {
                    grads.register::<B>(node.id, grad_bias);
                }
                if let Some(node) = node_lhs {
                    grads.register::<B>(node.id, grad_lhs);
                }
                if let Some(node) = node_rhs {
                    grads.register::<B>(node.id, grad_rhs);
                }
            }
        }

        // Prepare a stateful operation with each variable node and corresponding graph.
        //
        // Each node can be fetched with `ops.parents` in the same order as defined here.
        match FusedMatmulAddReluBackward
            .prepare::<C>([lhs.node.clone(), rhs.node.clone(), bias.node.clone()])
            // Marks the operation as compute bound, meaning it will save its
            // state instead of recomputing itself during checkpointing
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                // When at least one node is tracked, we should register our backward step.

                // The state consists of what will be needed for this operation's backward pass.
                // Since we need the parents' outputs, we must checkpoint their ids to retrieve
                // their node output at the beginning of the backward pass. We can also save
                // utilitary data such as the bias shape. If we also need this operation's output,
                // we can either save it in the state or recompute it.
                // during the backward pass. Here we choose to save it in the state because it's a
                // compute bound operation.
                let lhs_state = prep.checkpoint(&lhs);
                let rhs_state = prep.checkpoint(&rhs);
                let bias_shape = bias.primitive.shape();

                let output = B::fused_matmul_add_relu(
                    lhs.primitive.clone(),
                    rhs.primitive.clone(),
                    bias.primitive,
                );

                let state = (lhs_state, rhs_state, output.clone(), bias_shape);

                prep.finish(state, output)
            }
            OpsKind::UnTracked(prep) => {
                // When no node is tracked, we can just compute the original operation without
                // keeping any state.
                let output = B::fused_matmul_add_relu(lhs.primitive, rhs.primitive, bias.primitive);
                prep.finish(output)
            }
        }
    }
}
```

The previous code is self-documented to make it clearer, but here is what it does in summary:

We define `fused_matmul_add_relu` within `Autodiff<B>`, allowing any autodiff-decorated backend to
benefit from our implementation. In an autodiff-decorated backend, the forward pass must still be
implemented. This is achieved using a comprehensive match statement block where computation is
delegated to the inner backend, while keeping track of a state. The state comprises any information
relevant to the backward pass, such as input and output tensors, along with the bias shape. When an
operation isn't tracked (meaning there won't be a backward pass for this specific operation in the
graph), storing a state becomes unnecessary, and we simply perform the forward computation.

The backward pass uses the gradient obtained from the preceding node in the computation graph. It
calculates the derivatives for `relu` (`relu_backward`), add (no operation is required here, as the
derivative is one), and `matmul` (another `matmul` with transposed inputs). This results in
gradients for both input tensors and the bias, which are registered for consumption by subsequent
operation nodes.

The only remaining part is to implement our autodiff-decorated backend trait for our JIT Backend.

```rust, ignore
impl<R: CubeRuntime, F: FloatElement, I: IntElement> AutodiffBackend
    for Autodiff<CubeBackend<R, F, I>>
{
}
```

## Conclusion

In this guide, we've implemented a fused kernel using the `cubecl` compiler frontend, enabling
execution on any GPU and any `cubecl` backend. By delving into the inner workings of both the JIT
backend and the autodiff backend, we've gained a deeper understanding of these systems.

While extending a backend may be harder than working with straightforward tensors, the benefits can
be worth it. This approach enables the crafting of custom models with greater control over
execution, which can potentially greatly enhance the performance of your models.

As we conclude this guide, we hope that you have gained insights into Burn's world of backend
extensions, and that it will help you to unleash the full potential of your projects.
