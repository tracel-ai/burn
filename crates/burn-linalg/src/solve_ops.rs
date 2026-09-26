//! CPU solve dispatch and its analytic backward pass.
use burn_core as burn;
use burn_core::backend::{
    Backend, DispatchDevice, TensorMetadata, backend_extension, tensor::FloatTensor,
};
use burn_std::reader::try_read_sync;

// Other backends retain the tensor implementation, without host transfers.
pub(crate) fn supports_device(device: &DispatchDevice) -> bool {
    match device {
        #[cfg(feature = "flex")]
        DispatchDevice::Flex(_) => true,
        #[cfg(feature = "ndarray")]
        DispatchDevice::NdArray(_) => true,
        #[cfg(feature = "autodiff")]
        DispatchDevice::Autodiff(device) => supports_device(device),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

#[backend_extension(
    Flex: cfg(feature = "flex"),
    NdArray: cfg(feature = "ndarray"),
    Autodiff: cfg(feature = "autodiff"),
)]
pub(crate) trait SolveOps: Backend {
    // Inputs have the same rank, with unexpanded broadcast batch dimensions.
    fn solve(a: FloatTensor<Self>, b: FloatTensor<Self>) -> FloatTensor<Self> {
        let device = a.device();
        let msg = "linalg::solve: failed to read CPU tensor data";
        let a = try_read_sync(Self::float_into_data(a))
            .expect(msg)
            .expect(msg);
        let b = try_read_sync(Self::float_into_data(b))
            .expect(msg)
            .expect(msg);
        Self::float_from_data(crate::solve_host::solve_host_data(a, b), &device)
    }
}

#[cfg(feature = "flex")]
impl SolveOps for burn_core::backend::Flex {}
#[cfg(feature = "ndarray")]
impl SolveOps for burn_core::backend::NdArray {}

#[cfg(feature = "autodiff")]
mod autodiff {
    use super::*;
    use burn_autodiff::{
        Autodiff,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind, broadcast_shape},
    };
    use burn_std::Shape;

    impl<B: Backend + SolveOps, C: CheckpointStrategy> SolveOps for Autodiff<B, C> {
        fn solve(a: FloatTensor<Self>, b: FloatTensor<Self>) -> FloatTensor<Self> {
            #[derive(Debug)]
            struct Solve;

            impl<B: Backend + SolveOps> Backward<B, 2> for Solve {
                type State = (FloatTensor<B>, FloatTensor<B>, Shape);

                fn backward(
                    self,
                    ops: Ops<Self::State, 2>,
                    grads: &mut Gradients,
                    _checkpointer: &mut Checkpointer,
                ) {
                    let (a, x, b_shape) = ops.state;
                    let a_shape = a.shape();
                    let rank = a_shape.num_dims();
                    let grad = grads.consume::<B>(&ops.node);
                    let [node_a, node_b] = ops.parents;
                    // Empty RHS matrices have identically zero derivatives.
                    // Some backends cannot multiply matrices with inner size zero.
                    if x.shape()[rank - 1] == 0 {
                        let device = a.device();
                        let dtype = a.dtype().into();
                        if let Some(node) = node_a {
                            grads.register::<B>(node.id, B::float_zeros(a_shape, &device, dtype));
                        }
                        if let Some(node) = node_b {
                            grads.register::<B>(node.id, B::float_zeros(b_shape, &device, dtype));
                        }
                        return;
                    }
                    // dB = A^-T dX; dA = -dB X^T. Sum broadcast batch axes
                    // only after forming the per-system gradients.
                    let grad_b = B::solve(B::float_swap_dims(a, rank - 2, rank - 1), grad);
                    if let Some(node) = node_a {
                        let grad_a = B::float_neg(B::float_matmul(
                            grad_b.clone(),
                            B::float_swap_dims(x, rank - 2, rank - 1),
                        ));
                        grads.register::<B>(node.id, broadcast_shape::<B>(grad_a, &a_shape));
                    }
                    if let Some(node) = node_b {
                        grads.register::<B>(node.id, broadcast_shape::<B>(grad_b, &b_shape));
                    }
                }
            }

            let prep = Solve
                .prepare::<C>([a.node(), b.node()])
                .compute_bound()
                .stateful();
            let b_shape = b.shape();
            let a = a.into_primitive();
            let b = b.into_primitive();
            match prep {
                OpsKind::Tracked(prep) => {
                    let x = B::solve(a.clone(), b);
                    prep.finish((a, x.clone(), b_shape), x)
                }
                OpsKind::UnTracked(prep) => prep.finish(B::solve(a, b)),
            }
        }
    }
}
