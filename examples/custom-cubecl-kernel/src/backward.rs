use crate::FloatTensor;

use super::{AutodiffBackend, Backend};
use burn::{
    backend::autodiff::{
        Autodiff, NodeId,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind, broadcast_shape},
    },
    tensor::{Shape, TensorMetadata},
};
use burn_cubecl::{CubeBackend, CubeRuntime, FloatElement, IntElement, element::BoolElement};

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> AutodiffBackend
    for Autodiff<CubeBackend<R, F, I, BT>>
{
}

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
            type State = (NodeId, NodeId, FloatTensor<B>, Shape);

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
