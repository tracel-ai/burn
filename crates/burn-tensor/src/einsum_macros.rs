//! Literal Einstein summation with equation planning during compilation.

/// Computes Einstein summation from a literal equation.
///
/// The equation is parsed, validated, and planned at compile time. Operand counts
/// and exact ranks are checked statically where the equation determines them.
/// Axis sizes, broadcasting, and ellipsis widths are checked at runtime.
/// Each operand expression is evaluated once and moved into the contraction;
/// pass `&tensor` or `tensor.clone()` to retain an input.
///
/// # Expansion
///
/// The shared equation plan determines diagonal positions, input permutations,
/// reduction stages, matrix groupings, and output ordering. The macro generates
/// the corresponding `permute -> reshape -> matmul -> reshape -> permute` chains
/// and multiplication/reduction calls directly, in left-to-right operand order.
/// The generated code performs no equation parsing or shape-independent planning
/// at runtime.
///
/// Tensor sizes supply reshape dimensions at runtime. Singleton broadcasting can
/// allow earlier contractions or change which axes belong in the matrix groups.
/// Empty dimensions use a
/// multiply/reduce branch to preserve backend support and gradient connections.
/// Ellipsis axes are planned as symbolic blocks whose widths bind to input ranks.
///
/// For example, `einsum!("ij,jk->ik", a, b)` checks both inputs have rank two,
/// aligns them as `[i, k, j]`, and emits a matrix product contracting axis 2 (`j`)
/// with fixed input/output permutations. Only the sizes of `i`, `j`, and `k`
/// and any broadcasting or empty-dimension branches depend on the input tensors.
/// The generated code includes a documentation annotation describing its lowering.
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
