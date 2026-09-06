//! Literal Einstein summation with equation validation during compilation.

/// Computes Einstein summation from a literal equation.
///
/// The equation is parsed and validated at compile time. Operand counts and
/// exact ranks are checked statically where the equation determines them.
/// Axis sizes, broadcasting, and ellipsis widths are checked at runtime.
/// Each operand expression is evaluated once and moved into the contraction;
/// pass `&tensor` or `tensor.clone()` to retain an input.
///
/// # Expansion
///
/// The macro emits parsed input/output label arrays, rank checks, operand
/// alignment, one contraction stage per operand pair, and output finalization.
/// It never reparses the equation at runtime. Alignment extracts diagonals,
/// adds singleton dimensions, and permutes axes. Each contraction reduces
/// axes no longer needed by later operands and uses either multiplication or
/// `permute -> reshape -> batched matmul -> reshape -> permute`, depending on
/// runtime shapes. The contraction order is left to right.
///
/// For example, `einsum!("ij,jk->ik", a, b)` checks both inputs have rank two,
/// aligns them as `[i, k, j]`, contracts `j`, and returns `Tensor<2>`.
/// The generated code also includes a documentation annotation describing
/// the equation and its execution stages.
///
/// Scalar results use shape `[1]`. With an output ellipsis of unknown width,
/// supply the output type (for example `let y: Tensor<3> = einsum!(...)`).
/// For dynamic equation strings, use [`Tensor::einsum`](crate::Tensor::einsum).
///
/// # Example
///
/// ```
/// use burn_tensor::{Tensor, einsum};
/// let device = Default::default();
/// let a = Tensor::<2>::from_floats([[1., 2.], [3., 4.]], &device);
/// let b = Tensor::<2>::from_floats([[5., 6.], [7., 8.]], &device);
/// let c = einsum!("ij,jk->ik", a, b);
/// assert_eq!(c.into_data().try_to_vec::<f32>().unwrap(), [19., 22., 43., 50.]);
/// ```
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
#[macro_export]
macro_rules! einsum {
    ($($tt:tt)*) => {
        $crate::__einsum!($crate, $($tt)*)
    };
}
