use crate::FloatTensor;

use super::{AutodiffBackend, Backend};
use burn::{
    backend::{
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
            grads::Gradients,
            ops::{broadcast_shape, Backward, Ops, OpsKind},
            Autodiff, NodeID,
        },
        wgpu::{FloatElement, IntElement, JitBackend, WgpuRuntime},
    },
    tensor::Shape,
};

impl<F: FloatElement, I: IntElement> AutodiffBackend for Autodiff<JitBackend<WgpuRuntime, F, I>> {}

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
                    grads.register::<B, D>(node.id, grad_bias);
                }
                if let Some(node) = node_lhs {
                    grads.register::<B, D>(node.id, grad_lhs);
                }
                if let Some(node) = node_rhs {
                    grads.register::<B, D>(node.id, grad_rhs);
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
