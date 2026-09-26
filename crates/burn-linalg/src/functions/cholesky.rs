use crate::{DType, Tensor};
use alloc::vec;
use burn_std::{FloatDType, Slice};

/// Computes the Cholesky decomposition of real symmetric positive-definite matrices.
///
/// The input and output have shape `[..., n, n]`. If `upper` is false, returns
/// a lower triangular factor `L` such that `A = L @ L.transpose()`. Otherwise,
/// returns an upper triangular factor `U` such that `A = U.transpose() @ U`.
/// The factor has a positive diagonal, and its unused triangle is zero.
///
/// Only the selected triangle of the input is used. Symmetry is assumed, not
/// checked; the other triangle may contain arbitrary values. Gradients, when
/// enabled, likewise depend only on the selected triangle.
///
/// The output preserves the input device and dtype. F16 and BF16 inputs are
/// computed in F32 and cast back. Small positive pivots are accepted without
/// an epsilon cutoff; no diagonal regularization is applied. Nearly singular
/// inputs can fail or give inaccurate results because of rounding.
///
/// # Performance
///
/// Flex CPU tensors without autodiff use a direct blocked factorization with
/// matrix-matrix updates. Supported CubeCL devices with fixed 32-lane subgroups
/// use tiled GPU kernels for F32 tensors without autodiff. These
/// perform one final synchronous read of the failure status. Other dtypes,
/// backends, and autodiff tensors use the portable tensor implementation, which
/// synchronously checks each pivot. The portable path is slower on GPUs.
///
/// # Panics
///
/// Panics if the input has fewer than two dimensions, is not square, or is
/// quantized. Also panics if any matrix has a non-positive or non-finite pivot
/// during factorization, including singular and indefinite matrices.
///
/// # Example
///
/// ```rust,no_run
/// use burn_core::tensor::Tensor;
/// use burn_linalg::cholesky;
///
/// let device = Default::default();
/// let a = Tensor::<2>::from_data([[4.0, 2.0], [2.0, 5.0]], &device);
/// let l = cholesky(a, false);
/// // L = [[2, 0], [1, 2]]
/// ```
pub fn cholesky<const D: usize>(mut a: Tensor<D>, upper: bool) -> Tensor<D> {
    assert!(
        D >= 2,
        "linalg::cholesky: input must have at least two dimensions"
    );
    let dims = a.dims();
    let n = dims[D - 1];
    assert_eq!(dims[D - 2], n, "linalg::cholesky: input must be square");
    let original_dtype = a.dtype();
    assert!(
        !matches!(original_dtype, DType::QFloat(_)),
        "linalg::cholesky: input must have a real floating point dtype"
    );

    if n == 0 || dims[..D - 2].contains(&0) {
        return a;
    }

    let needs_upcast = matches!(original_dtype, DType::F16 | DType::BF16);
    if needs_upcast {
        a = a.cast(FloatDType::F32);
    }
    #[cfg(feature = "flex")]
    {
        let device = a.device();
        if !device.is_autodiff()
            && matches!(
                device.as_dispatch(),
                burn_core::backend::DispatchDevice::Flex(_)
            )
            && matches!(a.dtype(), DType::F32 | DType::F64)
        {
            use crate::cholesky_cpu::CholeskyCpuOps;
            use burn_core::backend::Dispatch;
            let factor = Tensor::<D>::from_dispatch(<Dispatch as CholeskyCpuOps>::cholesky_cpu(
                a.into_dispatch(),
                upper,
            ));
            return if needs_upcast {
                factor.cast(original_dtype)
            } else {
                factor
            };
        }
    }

    #[cfg(any(
        feature = "wgpu",
        feature = "webgpu",
        feature = "vulkan",
        feature = "metal",
        feature = "cuda",
        feature = "rocm",
        feature = "cpu"
    ))]
    {
        let device = a.device();
        if !device.is_autodiff()
            && matches!(
                device.as_dispatch(),
                burn_core::backend::DispatchDevice::Cube(cube)
                    if crate::cholesky_gpu::supported(cube, n, dims[..D-2].iter().product())
            )
            && a.dtype() == DType::F32
        {
            use crate::cholesky_gpu::CholeskyGpuOps;
            use burn_core::backend::Dispatch;
            use burn_core::tensor::Int;
            let (factor, info) =
                <Dispatch as CholeskyGpuOps>::cholesky_gpu(a.into_dispatch(), upper);
            let info = Tensor::<2, Int>::from_dispatch(info).into_data();
            for pivot in info.iter::<i32>() {
                assert!(
                    pivot == 0,
                    "linalg::cholesky: input is not positive definite or has non-finite values (pivot {})",
                    pivot
                );
            }
            let factor = Tensor::<D>::from_dispatch(factor);
            return if needs_upcast {
                factor.cast(original_dtype)
            } else {
                factor
            };
        }
    }

    if upper {
        a = a.swap_dims(D - 2, D - 1);
    }

    let mut l = a.zeros_like();
    let mut slices = vec![Slice::full(); D];
    for j in 0..n {
        // Subtract the contribution of earlier columns from A[j.., j].
        // Keep both matrix dimensions so matmul handles all batches together.
        let mut column = a.clone().slice_dim(D - 2, j..).slice_dim(D - 1, j);
        if j > 0 {
            let previous = l.clone().slice_dim(D - 2, j..).slice_dim(D - 1, 0..j);
            let row = previous.clone().slice_dim(D - 2, 0);
            column = column - previous.matmul(row.swap_dims(D - 2, D - 1));
        }

        let pivot = column.clone().slice_dim(D - 2, 0);
        let valid = pivot
            .clone()
            .greater_scalar(0.0)
            .bool_and(pivot.clone().is_finite());
        assert!(
            valid.all().into_scalar::<bool>(),
            "linalg::cholesky: input is not positive definite or has non-finite values (pivot {})",
            j + 1
        );
        let diagonal = pivot.sqrt();
        slices[D - 2] = Slice::from(j);
        slices[D - 1] = Slice::from(j);
        l = l.slice_assign(&slices, diagonal.clone());

        if j + 1 < n {
            slices[D - 2] = Slice::from(j + 1..);
            l = l.slice_assign(&slices, column.slice_dim(D - 2, 1..) / diagonal);
        }
    }

    if upper {
        l = l.swap_dims(D - 2, D - 1);
    }
    if needs_upcast {
        l.cast(original_dtype)
    } else {
        l
    }
}
