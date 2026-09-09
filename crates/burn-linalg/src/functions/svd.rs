use crate::{DType, LinalgOps, Tensor, check::TensorCheck};
use burn_core::backend::Dispatch;
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
/// # Arguments
///
/// * `tensor` - The input tensor of shape `[..., m, n]`.
/// * `sweeps` - Upper bound on the number of QR sweeps per singular value.
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
/// - `sweeps` is zero.
/// - The QR iteration does not converge within the requested sweep budget.
///
/// # Performance Note
/// The computation is dispatched to the backend through
/// [`LinalgOps::svd`], which backends may override with a native or
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
/// - Singular values are sorted in descending order.
/// - Internal norms, rotations, and shifts use scaled formulations to reduce
///   overflow and underflow for extreme finite inputs.
///
/// # Example
/// ```rust,ignore
/// use burn_linalg::svd;
/// use burn::Tensor;
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
    assert!(sweeps > 0, "linalg::svd: sweeps must be greater than zero");

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

    // Empty matrix or batch: the reduced SVD is empty too. Skip the backend
    // pipeline, which cannot obtain an aligned typed slice from every empty
    // tensor representation.
    let empty_batch = dims[..D - 2].contains(&0);
    if m == 0 || n == 0 || empty_batch {
        let mut du = [1; D];
        du[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
        du[D - 2] = n_rows;
        du[D - 1] = n;
        let mut ds = [1; D1];
        if D1 >= 2 {
            ds[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
        }
        ds[D1 - 1] = n;
        let mut dv = [1; D];
        dv[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
        dv[D - 2] = n;
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

    // Dispatch to the backend extension through the runtime dispatch tensor.
    // may be overridden by a backend with a native or fused SVD; the default
    // implementation runs the reference host pipeline on the pulled data,
    // which keeps this deterministic and backend-independent. The backend
    // returns the factors already sorted, permuted and swapped; its
    // dims follow the orientation (swap -> u is [..., n, n], vt is [..., n, m]).
    let (u, s, vt) = <Dispatch as LinalgOps>::svd(a.into_dispatch(), sweeps, swap);
    let result = (
        Tensor::<D>::from_dispatch(u),
        Tensor::<D1>::from_dispatch(s),
        Tensor::<D>::from_dispatch(vt),
    );

    if needs_upcast {
        (
            result.0.cast(original_dtype),
            result.1.cast(original_dtype),
            result.2.cast(original_dtype),
        )
    } else {
        result
    }
}
