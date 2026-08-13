use crate::{DType, Tensor, check, check::TensorCheck};
use burn_std::FloatDType;

/// Computes the singular value decomposition of a square or rectangular matrix.
///
/// This function decomposes the input tensor A into three tensors `U`, `S`, `Vt`
/// such that `A = U @ diag(S) @ Vt`, where:
/// - `U` has orthonormal columns of shape `[..., m, k]`
/// - `S` contains the singular values of `A`, sorted in descending order,
///   of shape `[..., k]`
/// - `Vt` has orthonormal rows of shape `[..., k, n]`
///
/// with `k = min(m, n)` (reduced decomposition, matching `torch.linalg.svd`).
///
/// # Algorithm
///
/// Two stages, mirroring the LAPACK `gesvd` / `dbdsqr` structure:
///
/// 1. **Golub-Kahan bidiagonalization** using Householder reflections (`A = U1 B V1^T`).
/// 2. **Implicitly shifted bidiagonal QR iteration** (LAPACK `dbdsqr`) to diagonalize `B`.
///
/// The algorithm is backward stable to within a small multiple of machine precision.
/// Convergence typically requires ~2.5 QR sweeps per singular value on average, bounded
/// by the `sweeps` argument.
///
/// # Arguments
///
/// * `tensor` - The input tensor of shape `[..., m, n]`.
/// * `sweeps` - Upper bound on the number of QR sweeps (the algorithm
///   typically converges in ~2.5 sweeps per singular value on average).
///
/// # Returns
///
/// A tuple of three tensors `(U, S, Vt)`:
/// - `U`: `[..., m, k]` with orthonormal columns
/// - `S`: `[..., k]` singular values in descending order
/// - `Vt`: `[..., k, n]` with orthonormal rows
///
/// # Generic Parameters
///
/// - `D`: The number of dimensions of the input tensor.
/// - `D1`: Must be set to `D - 1` (the rank of the singular value tensor).
///
/// # Panics
///
/// This function will panic if the tensor checks fail:
/// - The input tensor has less than 2 dimensions (`D < 2`).
/// - The input is a quantized tensor with dtype `DType::QFloat`.
/// - The generic parameters do not satisfy `D - 1 == D1`.
/// - The input tensor requires gradients (SVD has no autodiff support yet; detach first).
///
/// # Performance Note
/// The computation is dispatched to the backend through
/// `FloatTensorOps::float_svd`, which backends may override with a native or
/// fused implementation (none ship one yet). The default implementation
/// runs the reference pipeline on the host over the tensor data
/// (`into_data` / `from_data`), which is deterministic and
/// backend-independent, but the bidiagonalization is O(m n^2) scalar math.
/// It is not competitive with tuned native libraries (e.g. cuSOLVER) for
/// large matrices.
///
/// # Numerical Behavior
/// - If the input tensor has dtype F16 or BF16, it is internally upcast to
///   F32 for the computation and cast back to the original dtype before
///   returning, like `det` and `lu`.
/// - Singular values are sorted in descending order; values at or below
///   `10 * eps * sigma_max` (machine epsilon of the dtype) are treated as
///   numerical zeros and returned as 0.
/// - All internal norms, rotations and shifts are scale-invariant
///   (LAPACK-style), so inputs with entries up to `f32::MAX` and down to
///   subnormals stay finite and accurate.
///
/// # Example
/// ```rust,ignore
/// use burn_tensor::linalg::svd;
/// use burn_tensor::Tensor;
///
/// fn example() {
///     let device = Default::default();
///     let tensor = Tensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);
///
///     // Compute the singular value decomposition
///     let (u, s, vt) = svd::<2, 1>(tensor, 10);
///
///     // A = U @ diag(S) @ Vt (within tolerance)
///     let recon = u.mul(s.unsqueeze_dim(0)).matmul(vt);
///     println!("{}", recon);
/// }
/// ```
pub fn svd<const D: usize, const D1: usize>(
    mut tensor: Tensor<D>,
    sweeps: usize,
) -> (Tensor<D>, Tensor<D1>, Tensor<D>) {
    let dims = tensor.dims();
    let original_dtype = tensor.dtype();
    let device = tensor.device();
    check!(TensorCheck::svd_input_tensor::<D, D1>(
        "linalg::svd",
        &dims,
        original_dtype
    ));
    if tensor.is_require_grad() {
        panic!("linalg::svd: gradients are not implemented; detach the input tensor first");
    }

    // Upcast f16/bf16 to f32 (same convention as `det`), cast back at the end.
    let needs_upcast = original_dtype == DType::F16 || original_dtype == DType::BF16;
    if needs_upcast {
        tensor = tensor.cast(FloatDType::F32);
    }

    // One-sided formulation requires m >= n; decompose A^T for wide matrices.
    let (n_rows, n_cols) = (dims[D - 2], dims[D - 1]);
    let (a, swap) = if n_rows >= n_cols {
        (tensor, false)
    } else {
        (tensor.transpose(), true)
    };
    let (m, n) = (n_rows.max(n_cols), n_rows.min(n_cols));

    // Empty matrix (a zero leading dimension): the reduced SVD is empty too.
    // Skip the pipeline (bidiagonalization would index out of bounds).
    if m == 0 || n == 0 {
        let mut du = [1; D];
        du[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
        du[D - 2] = n_rows;
        du[D - 1] = 0;
        let mut ds = [1; D1];
        if D1 >= 2 {
            ds[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
        }
        ds[D1 - 1] = 0;
        let mut dv = [1; D];
        dv[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
        dv[D - 2] = 0;
        dv[D - 1] = n_cols;
        let u_t = Tensor::<D>::empty(du, (&device, DType::F32));
        let s_t = Tensor::<D1>::empty(ds, (&device, DType::F32));
        let vt_t = Tensor::<D>::empty(dv, (&device, DType::F32));
        let (u_t, s_t, vt_t) = if needs_upcast || original_dtype == DType::F64 {
            (
                u_t.cast(original_dtype),
                s_t.cast(original_dtype),
                vt_t.cast(original_dtype),
            )
        } else {
            (u_t, s_t, vt_t)
        };
        return (u_t, s_t, vt_t);
    }

    // Flush any in-flight kernels on the device (e.g. a previous test that
    // never read its outputs): cubecl host reads can fail with "strides are
    // not supported" while other kernels are still queued. No-op on eager
    // backends such as ndarray.
    let _ = device.sync();
    // Dispatch to the backend through the bridge: `FloatTensorOps::float_svd`
    // may be overridden by a backend with a native or fused SVD; the default
    // implementation runs the reference host pipeline on the pulled data,
    // which keeps this deterministic and backend-independent. The backend
    // returns the factors already sorted, masked, permuted and swapped; its
    // dims follow the orientation (swap -> u is [..., n, n], vt is [..., m, n]).
    let (u, s, vt) = crate::ops::svd(a.primitive, sweeps, swap);
    let result = (
        Tensor::<D>::new(u),
        Tensor::<D1>::new(s),
        Tensor::<D>::new(vt),
    );

    let result = if needs_upcast {
        (
            result.0.cast(original_dtype),
            result.1.cast(original_dtype),
            result.2.cast(original_dtype),
        )
    } else {
        result
    };
    // Flush the output transfers; no-op on eager backends.
    let _ = device.sync();
    result
}
