use crate::backend::Backend;
use crate::check::TensorCheck;
use crate::{Tensor, check, linalg};
use burn_std::{DType, FloatDType};

/// Computes the determinant on the last two dimensions of the input tensor.
///
/// # Arguments
/// - `tensor` - The input tensor of shape `[..., N, N]`.
///
/// # Returns
/// - The determinant tensor of shape `[...]` where its rank is less than the
///   input tensor's rank by two.
///
/// # Generic Parameters
/// - `D`: The rank of the input tensor.
/// - `D1`: Must be set to `D - 1`.
/// - `D2`: Must be set to `D - 2`.
///
/// # Panics
/// This function will panic if:
/// - The generic parameters do not satisfy `D - 1 == D1`.
/// - The generic parameters do not satisfy `D - 2 == D2`.
/// - The input tensor rank `D` is less than 3.
/// - The last two dimensions of the input tensor are not equal.
///
/// # Performance Note
/// The determinant function relies on the LU decomposition function under the hood,
/// which is not fully optimized. It will not be as fast as highly tuned specialized
/// libraries, especially for very large matrices or large batch sizes.
///
/// # Example
/// ```rust,ignore
/// use burn::tensor::Tensor;
/// use burn::tensor::linalg;
///
/// fn example<B: Backend>() {
///     let device = Default::default();
///     let tensor = Tensor::<B, 3>::from_data([[[4.0, 3.0], [6.0, 3.0]]], &device);
///
///     // Compute determinant
///     let result = linalg::det::<B, 3, 2, 1>(tensor);
///
///     // Expected Output:
///     // result: [-6.0]
/// }
///
/// fn example2<B: Backend>() {
///     let device = Default::default();
///     let tensor = Tensor::<B, 3>::from_data(
///         [
///             [[1.0, 2.0], [3.0, 4.0]],   // det = -2
///             [[2.0, 0.0], [0.0, 3.0]],   // det = 6
///             [[5.0, 6.0], [7.0, 8.0]],   // det = -2
///         ],
///         &device,
///     );
///
///     // Compute determinant
///     let result = linalg::det::<B, 3, 2, 1>(tensor);
///
///     // Expected Output:
///     // result: [-2.0, 6.0, -2.0]
/// }
/// ```
pub fn det<B: Backend, const D: usize, const D1: usize, const D2: usize>(
    mut tensor: Tensor<B, D>,
) -> Tensor<B, D2> {
    // Check whether input tensor has valid shape to compute determinant
    let dims = tensor.dims();
    check!(TensorCheck::det::<D, D1, D2>(dims));

    // Update f16 to f32
    let original_dtype = tensor.dtype();
    let needs_upcast = original_dtype == DType::F16 || original_dtype == DType::BF16;
    let working_float_dtype: FloatDType;
    if needs_upcast {
        working_float_dtype = FloatDType::F32;
        tensor = tensor.cast(working_float_dtype);
    } else {
        working_float_dtype = original_dtype.into()
    };

    // Compute determinant for base cases (1x1, 2x2, and 3x3 matrices)
    let rank = D as isize;
    if dims[D - 1] == 1 {
        let det_tensor = tensor.squeeze_dims::<D2>(&[rank - 2, rank - 1]);
        if needs_upcast {
            return det_tensor.cast(original_dtype)
        }
        return det_tensor
    } else if dims[D - 1] == 2 {
        let a = tensor.clone().slice_dim(D - 2, 0).slice_dim(D - 1, 0);
        let b = tensor.clone().slice_dim(D - 2, 0).slice_dim(D - 1, 1);
        let c = tensor.clone().slice_dim(D - 2, 1).slice_dim(D - 1, 0);
        let d = tensor.clone().slice_dim(D - 2, 1).slice_dim(D - 1, 1);
        let det_tensor = (a * d - b * c).squeeze_dims::<D2>(&[rank - 2, rank - 1]);
        if needs_upcast {
            return det_tensor.cast(original_dtype)
        }
        return det_tensor
    } else if dims[D - 1] == 3 {
        let a = tensor.clone().slice_dim(D - 2, 0).slice_dim(D - 1, 0);
        let b = tensor.clone().slice_dim(D - 2, 0).slice_dim(D - 1, 1);
        let c = tensor.clone().slice_dim(D - 2, 0).slice_dim(D - 1, 2);
        let d = tensor.clone().slice_dim(D - 2, 1).slice_dim(D - 1, 0);
        let e = tensor.clone().slice_dim(D - 2, 1).slice_dim(D - 1, 1);
        let f = tensor.clone().slice_dim(D - 2, 1).slice_dim(D - 1, 2);
        let g = tensor.clone().slice_dim(D - 2, 2).slice_dim(D - 1, 0);
        let h = tensor.clone().slice_dim(D - 2, 2).slice_dim(D - 1, 1);
        let i = tensor.clone().slice_dim(D - 2, 2).slice_dim(D - 1, 2);
        let det_tensor = (a * (e.clone() * i.clone() - f.clone() * h.clone())
            - b * (d.clone() * i - f * g.clone())
            + c * (d * h - e * g)).squeeze_dims::<D2>(&[rank - 2, rank - 1]);
        if needs_upcast {
            return det_tensor.cast(original_dtype)
        }
        return det_tensor
    }

    // Compute determinant for general case
    // det(A) = det(P) * det(L) * det(U)
    // det(A) = det(P) * 1 * det(U)
    let (lu, pivots) = linalg::compute_lu_decomposition::<B, D, D1>(tensor.clone());

    // Compute the determinant of P
    let squeezed_pivots = pivots.squeeze_dim::<D1>(D - 1);
    let n_pivots = squeezed_pivots.dims()[D1 - 1] as i64;
    let range_1d: Tensor<B, 1> =
        Tensor::arange(0..n_pivots, &tensor.device()).cast(working_float_dtype);
    let mut reshape_dims = [1; D1];
    reshape_dims[D1 - 1] = n_pivots;
    let range = range_1d.reshape(reshape_dims);
    let expand_dims: [usize; D1] = squeezed_pivots.dims();
    let batched_range_tensor = range.expand(expand_dims);
    let n_row_swaps = squeezed_pivots
        .not_equal(batched_range_tensor)
        .int()
        .sum_dim(D1 - 1);
    let odd_mask = n_row_swaps.clone().remainder_scalar(2).equal_elem(1);
    let p_det = n_row_swaps
        .cast(working_float_dtype)
        .ones_like()
        .mask_fill(odd_mask, -1.0)
        .squeeze_dim(D1 - 1);

    // Compute the determinant of U
    let u_diag = linalg::diag::<B, D, D1, _>(lu);
    let mut u_det = u_diag.clone().prod_dim(D1 - 1).squeeze_dim(D1 - 1);
    let epsilon = match tensor.dtype() {
        DType::F32 => 1e-5,
        DType::F64 => 1e-10,
        _ => 1e-5,
    };
    let near_zero = u_diag.abs().lower_equal_elem(epsilon);
    let singular_mask = near_zero.any_dim(D1 - 1).squeeze_dim::<D2>(D1 - 1);
    u_det = u_det.mask_fill(singular_mask, 0.0);

    let final_det = p_det * u_det;

    // Cast back to original dtype
    if needs_upcast {
        final_det.cast(original_dtype)
    } else {
        final_det
    }
}
