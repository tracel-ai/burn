use burn_core as burn;
use burn_core::backend::{Backend, TensorMetadata, backend_extension, tensor::FloatTensor};
use burn_std::reader::try_read_sync;

/// Linear algebra operations supplied by a backend extension.
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
    LibTorch: cfg(feature = "tch"),
    Remote: cfg(feature = "remote"),
    Capture: cfg(feature = "capture"),
)]
pub trait LinalgOps: Backend {
    /// Computes a reduced singular value decomposition.
    #[allow(unused_variables)]
    fn svd(
        tensor: FloatTensor<Self>,
        sweeps: usize,
        swap: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let device = tensor.device();
        let msg = "SVD fallback failed to synchronously read tensor data";
        let data = try_read_sync(Self::float_into_data(tensor))
            .expect(msg)
            .expect(msg);
        let (u, s, vt) = crate::svd_host::svd_host_data(data, sweeps, swap)
            .unwrap_or_else(|err| panic!("SVD fallback failed: {err}"));

        (
            Self::float_from_data(u, &device),
            Self::float_from_data(s, &device),
            Self::float_from_data(vt, &device),
        )
    }
}

#[allow(unused_macros)]
macro_rules! impl_linalg_ops {
    ($backend:ty) => {
        impl LinalgOps for $backend {}
    };
}

#[cfg(feature = "flex")]
impl_linalg_ops!(burn_core::backend::Flex);
#[cfg(feature = "cubecl-backend")]
impl LinalgOps for burn_cubecl::CubeBackend {}
#[cfg(feature = "ndarray")]
impl_linalg_ops!(burn_core::backend::NdArray);
#[cfg(feature = "tch")]
impl_linalg_ops!(burn_core::backend::LibTorch);
#[cfg(feature = "router")]
impl<C: burn_router::RouterChannel> LinalgOps for burn_router::BackendRouter<C> {}

#[cfg(feature = "fusion")]
impl<B> LinalgOps for burn_fusion::Fusion<B>
where
    B: burn_fusion::FusionBackend + LinalgOps,
{
    fn svd(
        tensor: FloatTensor<Self>,
        sweeps: usize,
        swap: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        use alloc::vec;
        use burn_core::tensor::Shape;
        use burn_fusion::{
            ExecutionError, FusionBackend, FusionRuntime,
            custom::{
                CustomOpIr, HandleContainer, Operation, OperationIr, OperationOutput, ScalarIr,
                StreamId, TensorIr,
            },
        };

        #[derive(Debug)]
        struct Svd<B> {
            desc: CustomOpIr,
            _backend: core::marker::PhantomData<B>,
        }

        impl<B> Operation<B::FusionRuntime> for Svd<B>
        where
            B: FusionBackend + LinalgOps,
        {
            fn execute(
                &self,
                handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
            ) -> Result<(), ExecutionError> {
                let ([input], [u_ir, s_ir, vt_ir]) = self.desc.as_fixed();
                let [ScalarIr::UInt(sweeps), ScalarIr::Bool(swap)] = self.desc.scalars.as_slice()
                else {
                    panic!("Invalid scalar arguments for linalg::svd custom operation")
                };

                let input = handles.get_float_tensor::<B>(input);
                let (u, s, vt) = B::svd(input, *sweeps as usize, *swap);
                handles.register_float_tensor::<B>(&u_ir.id, u);
                handles.register_float_tensor::<B>(&s_ir.id, s);
                handles.register_float_tensor::<B>(&vt_ir.id, vt);

                Ok(())
            }
        }

        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let rank = tensor.shape.num_dims();
        let dims = (0..rank)
            .map(|index| tensor.shape[index])
            .collect::<Vec<_>>();
        let (m, n) = (dims[rank - 2], dims[rank - 1]);

        let mut u_dims = dims.clone();
        let mut vt_dims = dims.clone();
        if swap {
            u_dims[rank - 2] = n;
            vt_dims[rank - 2] = n;
            vt_dims[rank - 1] = m;
        } else {
            vt_dims[rank - 2] = n;
            vt_dims[rank - 1] = n;
        }
        let mut s_dims = dims[..rank - 2].to_vec();
        s_dims.push(n);

        let outputs = [
            TensorIr::uninit(client.create_empty_handle(), Shape::from(u_dims), dtype),
            TensorIr::uninit(client.create_empty_handle(), Shape::from(s_dims), dtype),
            TensorIr::uninit(client.create_empty_handle(), Shape::from(vt_dims), dtype),
        ];
        let desc = CustomOpIr::with_scalars(
            "linalg::svd",
            &[tensor.into_ir()],
            &outputs,
            vec![ScalarIr::UInt(sweeps as u64), ScalarIr::Bool(swap)],
        );
        let [u, s, vt] = client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                Svd::<B> {
                    desc,
                    _backend: core::marker::PhantomData,
                },
            )
            .outputs();

        (u, s, vt)
    }
}
