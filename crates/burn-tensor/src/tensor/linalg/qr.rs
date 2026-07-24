use crate::{Tensor, check, check::TensorCheck, linalg::l2_norm, s};
use alloc::vec;
use burn_std::Slice;

/// Computes the QR decomposition of a square or rectangular matrix using Householder reflections.
///
/// This function decomposes the input tensor A into two tensors Q, R
/// such that A = QR, where Q is an orthonormal matrix and R is an upper triangular matrix.
/// If `reduced` is true then it returns reduced Q, R.
/// The reduced QR decomposition agrees with the full QR decomposition when n_cols >= n_rows (wide matrix).
///
/// # Arguments
/// - `tensor` - The input tensor of shape `[..., n_rows, n_cols]`.
/// - `reduced` - The bool value
///
/// # Returns
/// A tuple of two tensors `(Q, R)`:
/// - `Q` - The orthonormal tensor of shape `[..., n_rows, n_rows]` in case of `n_cols >= n_rows` or `reduced=false` otherwise `[..., n_rows, n_cols]`
/// - `R` - The upper triangular tensor of shape `[..., n_rows, n_cols]` in case of `n_cols >= n_rows` or `reduced=false` otherwise `[..., n_cols, n_cols]`
///
/// # Generic Parameters
///
/// - `D`: The number of dimensions of the input tensor.
///
/// # Panics
/// This function will panic if the tensor checks fail:
/// - The input tensor has less than 2 dimensions (`D < 2`).
/// - The input is a quantized tensor with dtype `DType::QFloat`.
///
/// # Example
/// ```rust,ignore
/// use burn::tensor::Tensor;
/// use burn::backend::Flex;
/// use burn::tensor::linalg;
///
/// fn example() {
///     let device = Default::default();
///     let tensor = Tensor::<2>::from_data([[3.0, 2.0], [4.0, 6.0]], &device);
///
///     // Compute Q, R
///     let (q, r) = linalg::qr::<2>(tensor);
///
///     // Expected Output:
///     // q: [[-0.6, 0.8],
///     //     [-0.8, -0.6]]
///     //
///     // r: [[-5.0, -6.0],
///     //     [0.0, -2.0]]
/// }
/// ```
pub fn qr<const D: usize>(tensor: Tensor<D>, reduced: bool) -> (Tensor<D>, Tensor<D>) {
    let dims = tensor.dims();
    let original_dtype = tensor.dtype();
    check!(TensorCheck::qr_input_tensor::<D>(
        "linalg::qr",
        &dims,
        original_dtype
    ));

    let device = tensor.device();
    let (n_rows, n_cols) = (dims[D - 2], dims[D - 1]);

    let max_iters = n_rows.min(n_cols);
    let mut r = tensor.clone();

    // Create tensor for storing Q
    let identity: Tensor<2> = Tensor::eye(n_rows, &device);
    let mut reshape_dims = [1; D];
    reshape_dims[D - 2] = n_rows;
    reshape_dims[D - 1] = n_rows;
    let reshaped_identity = identity.reshape(reshape_dims);
    let mut expand_dims = [n_rows; D];
    expand_dims[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
    let mut q = reshaped_identity.expand(expand_dims);

    let mut slices = vec![Slice::full(); D];
    for i in 0..max_iters {
        let sub_tensor = r
            .clone()
            .slice_dim(D - 2, s![i..])
            .slice_dim(D - 1, s![i..]);
        let v = sub_tensor.clone().slice_dim(D - 1, 0..1);
        let v0 = v.clone().slice_dim(D - 2, s![0]);
        let zeros = v0.clone().zeros_like();
        let norm_v = l2_norm(v.clone().slice_dim(D - 2, s![..]), D - 2);

        // removing zeros from the sign
        let sign = -v0.clone().sign();
        let mask = sign.clone().is_close(zeros.clone(), None, None);
        let sign = sign.mask_fill(mask, -1.0);

        // if norm_v==0, the vector w has to be zero and no reflection is applied
        let u0 = v0.clone().sub(norm_v.clone().mul(sign.clone()));

        let mask = norm_v.clone().is_close(zeros.clone(), None, None);
        let mut tau = -u0.clone().div(norm_v.clone()).mul(sign.clone());
        tau = tau.clone().mask_fill(mask.clone(), 0.0);

        let e0 = v0.clone().mul_scalar(0.0).add_scalar(1.0);
        let mut w = v.clone().div(u0.clone());
        slices[D - 2] = s![0];
        w = w.slice_assign(&slices, e0.clone());
        w = w.clone().mask_fill(mask, 0.0);

        // a closure to calculate tau*(w,w^{T} A)
        let f = |a: Tensor<D>| -> Tensor<D> {
            let aw = a.matmul(w.clone()).mul(tau.clone());
            w.clone().matmul(aw.transpose())
        };

        slices[D - 2] = s![i..];
        slices[D - 1] = s![i..];
        r = r.slice_assign(&slices, sub_tensor.clone() - f(sub_tensor.transpose()));

        slices[D - 2] = Slice::full();
        let q_sub_tensor = q.clone().slice(&slices);
        q = q.slice_assign(&slices, q_sub_tensor.clone() - f(q_sub_tensor).transpose());

        slices[D - 1] = Slice::full();
    }
    if reduced & (n_rows > n_cols) {
        slices[D - 1] = s![0..n_cols];
        let result_q = q.clone().slice(&slices);
        slices[D - 2] = s![0..n_cols];
        let result_r = r.clone().slice(&slices);
        return (result_q, result_r);
    }
    (q, r)
}
