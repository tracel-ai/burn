//! Backend solve dispatch and its analytic backward pass.
use burn_core as burn;
#[cfg(any(feature = "flex", feature = "ndarray", feature = "autodiff"))]
use burn_core::backend::TensorMetadata;
#[cfg(any(feature = "flex", feature = "ndarray"))]
use burn_core::backend::ops::FloatTensorOps;
use burn_core::backend::{Backend, DispatchDevice, backend_extension, tensor::FloatTensor};
#[cfg(any(feature = "flex", feature = "ndarray"))]
use burn_std::reader::try_read_sync;

// Other backends retain the tensor implementation.
pub(crate) fn supports_device(device: &DispatchDevice) -> bool {
    match device {
        #[cfg(feature = "flex")]
        DispatchDevice::Flex(_) => true,
        #[cfg(feature = "ndarray")]
        DispatchDevice::NdArray(_) => true,
        #[cfg(any(
            feature = "wgpu",
            feature = "webgpu",
            feature = "vulkan",
            feature = "metal",
            feature = "cuda",
            feature = "rocm",
            feature = "cpu"
        ))]
        DispatchDevice::Cube(_) => true,
        #[cfg(feature = "autodiff")]
        DispatchDevice::Autodiff(device) => supports_device(device),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

#[backend_extension(
    Flex: cfg(feature = "flex"),
    Cube: cfg(any(
        feature = "wgpu",
        feature = "webgpu",
        feature = "vulkan",
        feature = "metal",
        feature = "cuda",
        feature = "rocm",
        feature = "cpu"
    )),
    NdArray: cfg(feature = "ndarray"),
    Autodiff: cfg(feature = "autodiff"),
)]
pub(crate) trait SolveOps: Backend {
    // Inputs have the same rank, with unexpanded broadcast batch dimensions.
    fn solve(a: FloatTensor<Self>, b: FloatTensor<Self>) -> FloatTensor<Self>;
}

#[cfg(any(feature = "flex", feature = "ndarray"))]
macro_rules! impl_solve_host {
    ($backend:ty) => {
        impl SolveOps for $backend {
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
    };
}

#[cfg(feature = "flex")]
impl_solve_host!(burn_core::backend::Flex);
#[cfg(feature = "ndarray")]
impl_solve_host!(burn_core::backend::NdArray);

#[cfg(feature = "cubecl-backend")]
impl SolveOps for burn_cubecl::CubeBackend {
    fn solve(a: FloatTensor<Self>, b: FloatTensor<Self>) -> FloatTensor<Self> {
        crate::solve_cubecl::solve(a, b)
    }
}

#[cfg(feature = "fusion")]
impl<B> SolveOps for burn_fusion::Fusion<B>
where
    B: burn_fusion::FusionBackend + SolveOps,
{
    fn solve(a: FloatTensor<Self>, b: FloatTensor<Self>) -> FloatTensor<Self> {
        use burn_fusion::{
            ExecutionError, FusionBackend, FusionRuntime,
            custom::{
                CustomOpIr, HandleContainer, Operation, OperationIr, OperationOutput, StreamId,
                TensorIr,
            },
        };

        #[derive(Debug)]
        struct Solve<B> {
            desc: CustomOpIr,
            _backend: core::marker::PhantomData<B>,
        }

        impl<B: FusionBackend + SolveOps> Operation<B::FusionRuntime> for Solve<B> {
            fn execute(
                &self,
                handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
            ) -> Result<(), ExecutionError> {
                let ([a, b], [output]) = self.desc.as_fixed();
                let a = handles.get_float_tensor::<B>(a);
                let b = handles.get_float_tensor::<B>(b);
                handles.register_float_tensor::<B>(&output.id, B::solve(a, b));
                Ok(())
            }
        }

        let client = a.client.clone();
        let mut shape = b.shape.clone();
        for dim in 0..shape.num_dims() - 2 {
            if shape[dim] == 1 {
                shape[dim] = a.shape[dim];
            }
        }
        let outputs = [TensorIr::uninit(
            client.create_empty_handle(),
            shape,
            a.dtype,
        )];
        let desc = CustomOpIr::new("linalg::solve", &[a.into_ir(), b.into_ir()], &outputs);
        let [output] = client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                Solve::<B> {
                    desc,
                    _backend: core::marker::PhantomData,
                },
            )
            .outputs();
        // Resolve the handle to execute the operation and surface singularity
        // errors before returning, without reading the solution back to the host.
        let _ = client.resolve_tensor_float::<B>(output.clone());
        output
    }
}

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
