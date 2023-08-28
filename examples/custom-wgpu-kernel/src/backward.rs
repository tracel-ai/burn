use super::{CustomADBackend, CustomBackend};
use burn::tensor::{backend::Backend, ops::TensorOps, Shape};
use burn_autodiff::{
    grads::Gradients,
    ops::{broadcast_shape, Backward, Ops, OpsKind},
    ADBackendDecorator,
};
use burn_wgpu::{FloatElement, GraphicsApi, IntElement, WgpuBackend};

impl<G: GraphicsApi, F: FloatElement, I: IntElement> CustomADBackend
    for ADBackendDecorator<WgpuBackend<G, F, I>>
{
}

impl<G: GraphicsApi, F: FloatElement, I: IntElement> CustomBackend
    for ADBackendDecorator<WgpuBackend<G, F, I>>
{
    fn fused_matmul_add_relu<const D: usize>(
        lhs: <Self as Backend>::TensorPrimitive<D>,
        rhs: <Self as Backend>::TensorPrimitive<D>,
        bias: <Self as Backend>::TensorPrimitive<D>,
    ) -> <Self as Backend>::TensorPrimitive<D> {
        #[derive(Debug)]
        struct FusedMatmulAddReluBackward<const D: usize>;

        impl<B: CustomBackend, const D: usize> Backward<B, D, 3> for FusedMatmulAddReluBackward<D> {
            type State = (
                B::TensorPrimitive<D>,
                B::TensorPrimitive<D>,
                B::TensorPrimitive<D>,
                Shape<D>,
            );

            fn backward(self, ops: Ops<Self::State, 3>, grads: &mut Gradients) {
                let [node_lhs, node_rhs, node_bias] = ops.parents;
                let grad = grads.consume::<B, D>(&ops.node);

                let (lhs, rhs, output, shape_bias) = ops.state;

                let shape_lhs = B::shape(&lhs);
                let shape_rhs = B::shape(&rhs);
                let grad_output = B::relu_backward(output, grad);

                let grad_lhs = broadcast_shape::<B, D>(
                    B::matmul(grad_output.clone(), B::transpose(rhs)),
                    &shape_lhs,
                );
                let grad_rhs = broadcast_shape::<B, D>(
                    B::matmul(B::transpose(lhs), grad_output.clone()),
                    &shape_rhs,
                );
                let grad_bias = broadcast_shape::<B, D>(grad_output, &shape_bias);

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

        match FusedMatmulAddReluBackward
            .prepare(
                [lhs.node, rhs.node, bias.node],
                [lhs.graph, rhs.graph, bias.graph],
            )
            .statefull()
        {
            OpsKind::Tracked(prep) => {
                let bias_shape = WgpuBackend::<G, F, I>::shape(&bias.primitive);
                let output = WgpuBackend::<G, F, I>::fused_matmul_add_relu(
                    lhs.primitive.clone(),
                    rhs.primitive.clone(),
                    bias.primitive,
                );

                let state = (lhs.primitive, rhs.primitive, output.clone(), bias_shape);
                prep.finish(state, output)
            }
            OpsKind::UnTracked(prep) => {
                let output = WgpuBackend::<G, F, I>::fused_matmul_add_relu(
                    lhs.primitive,
                    rhs.primitive,
                    bias.primitive,
                );
                prep.finish(output)
            }
        }
    }
}
