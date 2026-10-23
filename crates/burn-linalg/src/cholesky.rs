#[cfg(feature = "fusion")]
use alloc::{vec, vec::Vec};
use burn_core as burn;
use burn_core::backend::{
    Backend, backend_extension,
    tensor::{FloatTensor, IntTensor},
};
#[cfg(feature = "fusion")]
use burn_std::{DType, Shape};

#[backend_extension(
    Flex: cfg(feature = "flex"),
    Cube: cfg(any(feature = "wgpu", feature = "webgpu", feature = "vulkan",
        feature = "metal", feature = "cuda", feature = "rocm", feature = "cpu")),
    Fusion: cfg(feature = "fusion")
)]
pub(crate) trait CholeskyOps: Backend {
    /// Returns the factor and an I32 status tensor of shape `[batch]` on the input device.
    /// Each status is zero on success, otherwise the first invalid pivot's one-based index.
    /// Inputs must be nonempty square matrices with a supported, promoted dtype.
    #[fusion(meta = |tensor, _upper| {
        let n = tensor.shape[tensor.shape.num_dims() - 1];
        let batch = tensor.shape.num_elements() / (n * n);
        (tensor.clone(), burn::backend::fusion::custom::TensorSpec::new(
            Shape::new([batch]), DType::I32))
    })]
    fn cholesky(tensor: FloatTensor<Self>, upper: bool) -> (FloatTensor<Self>, IntTensor<Self>);
}

#[cfg(all(test, feature = "flex"))]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
    use burn_core::backend::Dispatch;
    use burn_core::tensor::{DType, Device, Int, Tensor, TensorData};

    #[test]
    fn reports_first_invalid_pivot_per_matrix() {
        let device = Device::default();
        let n = 65;
        let batch = 4;
        #[cfg(any(
            feature = "wgpu",
            feature = "webgpu",
            feature = "vulkan",
            feature = "metal",
            feature = "cuda",
            feature = "rocm",
            feature = "cpu"
        ))]
        if let burn_core::backend::DispatchDevice::Cube(cube) = device.as_dispatch()
            && !crate::cholesky_gpu::supported(cube, n, batch)
        {
            // Unsupported devices use the portable public API path.
            return;
        }
        let mut matrices = vec![0.0f32; batch * n * n];
        for b in 0..batch {
            for i in 0..n {
                matrices[b * n * n + i * n + i] = 4.0;
            }
        }
        matrices[n * n + 16 * n + 16] = -1.0;
        matrices[n * n + 64 * n + 64] = -1.0;
        matrices[2 * n * n + 64 * n + 64] = -1.0;
        for upper in [false, true] {
            let input =
                Tensor::<3>::from_data(TensorData::new(matrices.clone(), [batch, n, n]), &device);
            let (_, info) = Dispatch::cholesky(input.into_dispatch(), upper);
            let info = Tensor::<1, Int>::from_dispatch(info);
            assert_eq!(info.dims(), [batch]);
            assert_eq!(info.dtype(), DType::I32);
            assert_eq!(
                info.into_data().iter::<i32>().collect::<Vec<_>>(),
                vec![0, 17, 65, 0]
            );
        }
    }
}
