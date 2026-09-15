//! Literal Einstein summation with equation planning during compilation.

/// Computes Einstein summation from a literal equation.
///
/// Give each input axis a letter, separate operands with commas, and use `->`
/// to specify the output axes and their order. Values are multiplied along
/// matching labels and summed over labels omitted from the output. For example,
/// `"ij,jk->ik"` multiplies two matrices and sums over their shared `j` axis.
///
/// | Equation | Operation |
/// | --- | --- |
/// | `"ij,jk->ik"` | Matrix multiplication |
/// | `"ij->ji"` | Transpose |
/// | `"ii->i"` | Extract a diagonal |
/// | `"ii->"` | Trace (sum of the diagonal) |
/// | `"i,i->"` | Dot product |
/// | `"...ij,...jk->...ik"` | Matrix multiplication with broadcast batch dimensions |
///
/// For dynamic equation strings, use [`Tensor::einsum`](crate::Tensor::einsum).
///
/// # Examples
///
/// ```
/// use burn_tensor::{Tensor, einsum};
/// let device = Default::default();
/// let a = Tensor::<2>::from_floats([[1., 2.], [3., 4.]], &device);
/// let b = Tensor::<2>::from_floats([[5., 6.], [7., 8.]], &device);
/// let c = einsum!("ij,jk->ik", &a, b);
/// assert_eq!(c.into_data().try_to_vec::<f32>().unwrap(), [19., 22., 43., 50.]);
///
/// let transposed = einsum!("ij->ji", &a);
/// assert_eq!(transposed.into_data().try_to_vec::<f32>().unwrap(), [1., 3., 2., 4.]);
/// let trace = einsum!("ii->", a);
/// assert_eq!(trace.dims(), [1]);
/// assert_eq!(trace.into_data().try_to_vec::<f32>().unwrap(), [5.]);
/// ```
///
/// # Broadcasting and equation rules
///
/// Matching labels across operands must have equal sizes or a size of one, which
/// broadcasts to the other size. Repeating a label within one operand extracts
/// its diagonal: those axis sizes must be equal, even if one is a singleton.
///
/// `...` matches zero or more axes. Ellipsis dimensions broadcast from the right,
/// allowing operands to have different numbers of batch dimensions. To sum over
/// these dimensions, omit `...` from the explicit output.
///
/// ```
/// use burn_tensor::{Tensor, einsum};
/// let device = Default::default();
/// let a = Tensor::<3>::ones([2, 3, 4], &device);
/// let b = Tensor::<2>::ones([4, 5], &device);
/// // b is shared across both batches of a.
/// let c: Tensor<3> = einsum!("...ij,...jk->...ik", a, b);
/// assert_eq!(c.dims(), [2, 3, 5]);
/// ```
///
/// Each axis label is a single letter from `a-z` or `A-Z`. Uppercase and lowercase
/// letters are distinct labels. Spaces between tokens are ignored.
/// With no `->`, the output contains the ellipsis first, followed by labels
/// occurring exactly once across all inputs, sorted `A-Z`, then `a-z`.
/// For example, `"ij,jk"` is equivalent to `"ij,jk->ik"`.
///
/// # Types and limitations
///
/// Float and Int operands must share their kind, dtype, and device. Float
/// operations support automatic differentiation; quantized operands are unsupported.
/// Scalar results have shape `[1]`, and an empty input subscript accepts shape `[1]`.
/// With an output ellipsis of unknown width, supply the output type, as above.
/// Operands are contracted from left to right; no optimized contraction order is searched.
///
/// # Validation
///
/// Equations, operand counts, and exact ranks are checked at compile time where
/// the equation determines them. Axis sizes and ellipsis widths are checked at runtime.
///
/// Unknown output labels are compile errors:
///
/// ```compile_fail
/// use burn_tensor::{Tensor, einsum};
/// let a = Tensor::<1>::zeros([3], &Default::default());
/// let _ = einsum!("i->j", a);
/// ```
///
/// So are incorrect input ranks:
///
/// ```compile_fail
/// use burn_tensor::{Tensor, einsum};
/// let a = Tensor::<1>::zeros([3], &Default::default());
/// let _ = einsum!("ij->ji", a);
/// ```
///
/// And incorrect output ranks:
///
/// ```compile_fail
/// use burn_tensor::{Tensor, einsum};
/// let a = Tensor::<2>::zeros([2, 3], &Default::default());
/// let _: Tensor<1> = einsum!("ij->ji", a);
/// ```
/// An equation must have one subscript per operand:
///
/// ```compile_fail
/// use burn_tensor::{Tensor, einsum};
/// let a = Tensor::<1>::zeros([3], &Default::default());
/// let _ = einsum!("i,j->ij", a);
/// ```
///
/// Malformed ellipses are rejected during compilation:
///
/// ```compile_fail
/// use burn_tensor::{Tensor, einsum};
/// let a = Tensor::<1>::zeros([3], &Default::default());
/// let _ = einsum!("i..->i", a);
/// ```
///
/// # Panics
///
/// Panics for incompatible ranks or broadcast dimensions, unequal repeated-label
/// dimensions, mismatched devices or dtypes, quantized operands, or an incorrect
/// output rank when the output contains an ellipsis of unknown width.
#[macro_export]
macro_rules! einsum {
    ($($tt:tt)*) => {
        $crate::__einsum!($crate, $($tt)*)
    };
}
