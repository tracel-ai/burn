# Custom WGPU Kernel

In this section, you will learn how to create your own custom operation by writing your own kernel
with the WGPU backend. We will take the example of a common workflow in the deep learning field,
where we create a kernel to fuse multiple operations together. We will fuse a matmul kernel followed
by an addition and the ReLU activation function, which is commonly found in various models. All the
code can be found under the
[examples directory](https://github.com/tracel-ai/burn/tree/main/examples/custom-wgpu-kernel).

## Custom Backend Trait

First, we need to determine the type signature of our newly created operation by defining our custom
backend traits. As we will use the associated type `TensorPrimitive` of the `Backend` trait, which
encapsulates the underlying tensor implementation of the backend, we will use a type alias to avoid
the ugly disambiguation with associated types.

```rust, ignore
/// We use a type alias for better readability.
pub type FloatTensor<B, const D: usize> = <B as burn::tensor::backend::Backend>::TensorPrimitive<D>;

/// We create our own Backend trait that extends the Burn backend trait.
pub trait Backend: burn::tensor::backend::Backend {
    fn fused_matmul_add_relu<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
        bias: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D>;
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
        lhs.into_primitive(),
        rhs.into_primitive(),
        bias.into_primitive(),
    );

    Tensor::from_primitive(output)
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

Now, let's proceed to write the fused kernel using the WGSL shading language. To keep things simple,
we'll create a straightforward matmul kernel without employing any intricate techniques. Although we
won't delve into the details of the WGSL syntax, as it falls beyond the scope of this guide, we
still provide the implementation below for readers who are curious. The actual matmul, add and relu
computations are found at the end, after an extensive overhead whose use is to correctly map each
thread to the data it is responsible of, with support for batches.

```wgsl, ignore
@group(0)
@binding(0)
var<storage, read> lhs: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> rhs: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read> bias: array<{{ elem }}>;

@group(0)
@binding(3)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(4)
var<storage, read> info: array<u32>;

const BLOCK_SIZE = {{ workgroup_size_x }}u;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    // Indices
    let row = workgroup_id.x * BLOCK_SIZE + (local_idx / BLOCK_SIZE);
    let col = workgroup_id.y * BLOCK_SIZE + (local_idx % BLOCK_SIZE);
    let batch = global_id.z;

    // Basic information
    let dim = info[0];
    let n_rows = info[6u * dim - 1u];
    let n_cols = info[6u * dim];
    let K = info[5u * dim - 1u];

    // Returns if outside the output dimension
    if row >= n_rows || col >= n_cols {
        return;
    }

    // Calculate the corresponding offsets with support for broadcasting.
    let offset_output = batch * n_rows * n_cols;
    var offset_lhs: u32 = 0u;
    var offset_rhs: u32 = 0u;

    let batch_dims = dim - 2u;
    for (var b: u32 = 1u; b <= batch_dims; b++) {
        let stride_lhs = info[b];
        let stride_rhs = info[b + dim];
        let stride_output = info[b + 2u * dim];
        let shape_lhs = info[b + 3u * dim];
        let shape_rhs = info[b + 4u * dim];

        offset_lhs += offset_output / stride_output % shape_lhs * stride_lhs;
        offset_rhs += offset_output / stride_output % shape_rhs * stride_rhs;
    }

    // Basic matmul implementation
    var sum = 0.0;
    for (var k: u32 = 0u; k < K; k++) {
        let lhs_index = row * K + k;
        let rhs_index = k * n_cols + col;

        sum += lhs[offset_lhs + lhs_index] * rhs[offset_rhs + rhs_index];
    }

    let output_index = row * n_cols + col;
    let index = offset_output + output_index;

    // Add and ReLU
    output[index] = max(sum + bias[index], 0.0);
}
```

Now, let's move on to the next step, which involves implementing the remaining code to launch the
kernel. The initial part entails loading the template and populating it with the appropriate
variables. The `register(name, value)` method simply replaces occurrences of `{{ name }}` in the
above WGSL code with some other string before it is compilated. In order to use templating
utilities, you will have to activate the `template` feature of Burn in your `cargo.toml`.

```rust, ignore
// Source the kernel written in WGSL.
kernel_wgsl!(FusedMatmulAddReluRaw, "./kernel.wgsl");

// Define our kernel type with cube information.
#[derive(new, Debug)]
struct FusedMatmulAddRelu<E: FloatElement> {
    cube_dim: CubeDim,
    _elem: PhantomData<E>,
}

// Implement the dynamic kernel trait for our kernel type.
impl<E: FloatElement> KernelSource for FusedMatmulAddRelu<E> {
    fn source(&self) -> SourceTemplate {
        // Extend our raw kernel with workgroup size information using the
        // `SourceTemplate` trait.
        FusedMatmulAddReluRaw::new()
            .source()
            .register("workgroup_size_x", self.cube_dim.x.to_string())
            .register("workgroup_size_y", self.cube_dim.y.to_string())
            .register("elem", E::type_name())
            .register("int", "i32")
    }
}
```

Subsequently, we'll go into implementing our custom backend trait for the WGPU backend. Note that we
won't go into supporting the `fusion` feature flag in this tutorial, so we implement the trait for
the raw `WgpuBackend` type.

```rust, ignore
/// Implement our custom backend trait for the existing backend `WgpuBackend`.
impl<F: FloatElement, I: IntElement> Backend for JitBackend<WgpuRuntime, F, I> {
    fn fused_matmul_add_relu<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
        bias: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        // Define cube dim, hardcoded for simplicity.
        let cube_dim = CubeDim { x: 16, y: 16, z: 1 };

        lhs.assert_is_on_same_device(&rhs);
        lhs.assert_is_on_same_device(&bias);

        // For simplicity, make sure each tensor is continuous.
        let lhs = into_contiguous(lhs);
        let rhs = into_contiguous(rhs);
        let bias = into_contiguous(bias);

        // Get the matmul relevant shapes.
        let num_rows = lhs.shape.dims[D - 2];
        let num_cols = rhs.shape.dims[D - 1];

        // Compute shape of output, while tracking number of batches.
        let mut num_batches = 1;
        let mut shape_out = [0; D];
        for i in shape_out.into_iter().take(D - 2) {
            shape_out[i] = usize::max(lhs.shape.dims[i], rhs.shape.dims[i]);
            num_batches *= shape_out[i];
        }
        shape_out[D - 2] = num_rows;
        shape_out[D - 1] = num_cols;
        let shape_out = Shape::new(shape_out);

        // Create a buffer for the output tensor.
        let buffer = lhs
            .client
            .empty(shape_out.num_elements() * core::mem::size_of::<F>());

        // Create the output tensor primitive.
        let output = JitTensor::new(lhs.client.clone(), lhs.device.clone(), shape_out, buffer);

        // Create the kernel.
        let kernel = FusedMatmulAddRelu::<F>::new(cube_dim);

        // Build info buffer with tensor information needed by the kernel, such as shapes and strides.
        let info = build_info(&[&lhs, &rhs, &output]);
        let info_handle = lhs.client.create(bytemuck::cast_slice(&info));

        // Declare the wgsl workgroup with the number of blocks in x, y and z.
        let cubes_needed_in_x = f32::ceil(num_rows as f32 / cube_dim.x as f32) as u32;
        let cubes_needed_in_y = f32::ceil(num_cols as f32 / cube_dim.y as f32) as u32;
        let cube_count = CubeCount::new(cubes_needed_in_x, cubes_needed_in_y, num_batches as u32);

        // Execute lazily the kernel with the launch information and the given buffers.
        lhs.client.execute(
            Box::new(SourceKernel::new(kernel, cube_count, cube_dim)),
            vec![
                lhs.handle.binding(),
                rhs.handle.binding(),
                bias.handle.binding(),
                output.handle.clone().binding(),
                info_handle.binding(),
            ],
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

Now that the custom backend trait is implemented for the WGPU backend, you can use it to invoke the
`matmul_add_relu_custom` function. However, calculating gradients is not yet possible at this stage.
If your use case does not extend beyond inference, there is no need to implement any of the
following code.

For the backward pass, we will leverage the backend implementation from `burn-autodiff`, which is
actually generic over the backend. Instead of crafting our own WGSL kernel for the backward pass, we
will use our fused kernel only for the forward pass, and compute the gradient using basic
operations.

```rust, ignore
// Implement our custom backend trait for any backend that also implements our custom backend trait.
//
// Note that we could implement the backend trait only for the Wgpu backend instead of any backend that
// also implements our own API. This would allow us to call any function only implemented for Wgpu
// and potentially call a custom kernel crafted only for this task.
impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    fn fused_matmul_add_relu<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
        bias: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        // Create our zero-sized type that will implement the Backward trait.
        #[derive(Debug)]
        struct FusedMatmulAddReluBackward<const D: usize>;

        // Implement the backward trait for the given backend B, the node gradient being of rank D
        // with three other gradients to calculate (lhs, rhs, and bias).
        impl<B: Backend, const D: usize> Backward<B, D, 3> for FusedMatmulAddReluBackward<D> {
            // Our state that we must build during the forward pass to compute the backward pass.
            //
            // Note that we could improve the performance further by only keeping the state of
            // tensors that are tracked, improving memory management, but for simplicity, we avoid
            // that part.
            type State = (NodeID, NodeID, FloatTensor<B, D>, Shape<D>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                // Get the nodes of each variable.
                let [node_lhs, node_rhs, node_bias] = ops.parents;
                // Fetch the gradient for the current node.
                let grad = grads.consume::<B, D>(&ops.node);

                // Set our state.
                let (lhs_state, rhs_state, output, shape_bias) = ops.state;
                let lhs = checkpointer.retrieve_node_output(lhs_state);
                let rhs = checkpointer.retrieve_node_output(rhs_state);

                // Fetch shapes of our tensor to support broadcasting.
                let shape_lhs = B::float_shape(&lhs);
                let shape_rhs = B::float_shape(&rhs);

                // Compute the gradient of the output using the already existing `relu_backward`
                // function in the basic Burn backend trait.
                let grad_output = B::relu_backward(output, grad);

                // Compute the lhs gradient, which is the derivative of matmul with support for
                // broadcasting.
                let grad_lhs = broadcast_shape::<B, D>(
                    B::float_matmul(grad_output.clone(), B::float_transpose(rhs)),
                    &shape_lhs,
                );
                // Compute the rhs gradient, which is the derivative of matmul with support for
                // broadcasting.
                let grad_rhs = broadcast_shape::<B, D>(
                    B::float_matmul(B::float_transpose(lhs), grad_output.clone()),
                    &shape_rhs,
                );
                // The add derivative is only 1, so we just need to support broadcasting to
                // compute the bias gradient.
                let grad_bias = broadcast_shape::<B, D>(grad_output, &shape_bias);

                // Register the gradient for each variable based on whether they are marked as
                // `tracked`.
                if let Some(node) = node_bias {
                    grads.register::<B, D>(node, grad_bias);
                }
                if let Some(node) = node_lhs {
                    grads.register::<B, D>(node, grad_lhs);
                }
                if let Some(node) = node_rhs {
                    grads.register::<B, D>(node, grad_rhs);
                }
            }
        }

        // Prepare a stateful operation with each variable node and corresponding graph.
        //
        // Each node can be fetched with `ops.parents` in the same order as defined here.
        match FusedMatmulAddReluBackward
            .prepare::<C>(
                [lhs.node.clone(), rhs.node.clone(), bias.node.clone()],
                [lhs.graph.clone(), rhs.graph.clone(), bias.graph.clone()],
            )
            // Marks the operation as compute bound, meaning it will save its
            // state instead of recomputing itself during checkpointing
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                // When at least one node is tracked, we should register our backward step.

                // The state consists of what will be needed for this operation's backward pass.
                // Since we need the parents' outputs, we must checkpoint their ids to retrieve their node
                // output at the beginning of the backward. We can also save utilitary data such as the bias shape
                // If we also need this operation's output, we can either save it in the state or recompute it
                // during the backward pass. Here we choose to save it in the state because it's a compute bound operation.
                let lhs_state = prep.checkpoint(&lhs);
                let rhs_state = prep.checkpoint(&rhs);
                let bias_shape = B::float_shape(&bias.primitive);

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

The previous code is self-documented to make it clearer, but here is what it does in summary.

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

The only remaining part is to implement our autodiff-decorated backend trait for our WGPU Backend.

```rust, ignore
impl<G: GraphicsApi, F: FloatElement, I: IntElement> AutodiffBackend for Autodiff<WgpuBackend<G, F, I>>
{
}
```

## Conclusion

In this guide, we've implemented a fused kernel using the WGPU backend, enabling execution on any
GPU. By delving into the inner workings of both the WGPU backend and the autodiff backend, we've
gained a deeper understanding of these systems.

While extending a backend may be harder than working with straightforward tensors, the benefits can
be worth it. This approach enables the crafting of custom models with greater control over
execution, which can potentially greatly enhance the performance of your models.

It is worth noting that while the manual fusion of operations can be valuable, our future plans
include the development of a backend extension that will automate this process. As we conclude this
guide, we hope that you have gained insights into Burn's world of backend extensions, and that it
will help you to unleash the full potential of your projects.
