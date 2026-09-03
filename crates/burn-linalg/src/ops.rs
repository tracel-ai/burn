use burn_core as burn;
use burn_core::backend::{Backend, TensorMetadata, backend_extension, tensor::FloatTensor};
use burn_std::reader::try_read_sync;

/// Linear algebra operations supplied by a backend extension.
#[backend_extension(
    Flex: cfg(feature = "flex"),
    Wgpu: cfg(feature = "wgpu"),
    WebGpu: cfg(feature = "webgpu"),
    Vulkan: cfg(feature = "vulkan"),
    Metal: cfg(feature = "metal"),
    Cuda: cfg(feature = "cuda"),
    Rocm: cfg(feature = "rocm"),
    Cpu: cfg(feature = "cpu"),
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
impl<R: burn_cubecl::CubeRuntime> LinalgOps for burn_cubecl::CubeBackend<R> {}
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
        let client = tensor.client.clone();
        let resolved = client.resolve_tensor_float::<B>(tensor);
        let (u, s, vt) = B::svd(resolved, sweeps, swap);
        (
            burn_fusion::register_float_tensor::<B>(u, &client),
            burn_fusion::register_float_tensor::<B>(s, &client),
            burn_fusion::register_float_tensor::<B>(vt, &client),
        )
    }
}
