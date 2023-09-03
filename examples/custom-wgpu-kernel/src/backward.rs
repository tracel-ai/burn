use crate::FloatTensor;

use super::{ADBackend, Backend};
use burn::autodiff::{
    grads::Gradients,
    ops::{broadcast_shape, Backward, Ops, OpsKind},
    ADBackendDecorator,
};
use burn::backend::wgpu::{FloatElement, GraphicsApi, IntElement, WgpuBackend};
use burn::tensor::Shape;

impl<G: GraphicsApi, F: FloatElement, I: IntElement> ADBackend
    for ADBackendDecorator<WgpuBackend<G, F, I>>
{
}

// Implement our custom backend trait for any backend that also implements our custom backend trait.
//
// Note that we could implement the backend trait only for the Wgpu backend instead of any backend that
// also implements our own API. This would allow us to call any function only implemented for Wgpu
// and potentially call a custom kernel crafted only for this task.
impl<B: Backend> Backend for ADBackendDecorator<B> {
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
            type State = (
                FloatTensor<B, D>,
                FloatTensor<B, D>,
                FloatTensor<B, D>,
                Shape<D>,
            );

            fn backward(self, ops: Ops<Self::State, 3>, grads: &mut Gradients) {
                // Get the nodes of each variable.
                let [node_lhs, node_rhs, node_bias] = ops.parents;
                // Fetch the gradient for the current node.
                let grad = grads.consume::<B, D>(&ops.node);

                // Set our state.
                let (lhs, rhs, output, shape_bias) = ops.state;

                // Fetch shapes of our tensor to support broadcasting.
                let shape_lhs = B::shape(&lhs);
                let shape_rhs = B::shape(&rhs);

                // Compute the gradient of the output using the already existing `relu_backward`
                // function in the basic Burn backend trait.
                let grad_output = B::relu_backward(output, grad);

                // Compute the lhs gradient, which is the derivative of matmul with support for
                // broadcasting.
                let grad_lhs = broadcast_shape::<B, D>(
                    B::matmul(grad_output.clone(), B::transpose(rhs)),
                    &shape_lhs,
                );
                // Compute the rhs gradient, which is the derivative of matmul with support for
                // broadcasting.
                let grad_rhs = broadcast_shape::<B, D>(
                    B::matmul(B::transpose(lhs), grad_output.clone()),
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
            .prepare(
                [lhs.node, rhs.node, bias.node],
                [lhs.graph, rhs.graph, bias.graph],
            )
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                // When at least one node is tracked, we should register our backward step.
                // We compute the output and the state before finishing the preparation.
                let bias_shape = B::shape(&bias.primitive);
                let output = B::fused_matmul_add_relu(
                    lhs.primitive.clone(),
                    rhs.primitive.clone(),
                    bias.primitive,
                );

                let state = (lhs.primitive, rhs.primitive, output.clone(), bias_shape);
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
