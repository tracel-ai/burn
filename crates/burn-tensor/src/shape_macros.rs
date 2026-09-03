//! Shape assertion macros. Parsing and expansion live in burn-derive; these wrappers forward
//! `$crate` so the expansion can name [`Tensor::dims`](crate::Tensor::dims) by path, which
//! makes anything other than a `Tensor` a type error.
//!
//! Inspired by the `burn-contracts` crate by Crutcher Dunnavant.

/// Returns tensor axis sizes by position while checking the rest of the shape.
///
/// The pattern has one slot per axis. A bare identifier labels an axis whose size is returned:
/// the result is a tuple with one `usize` per label, in pattern order, and the label itself is
/// not bound. `=expr` checks the axis against an in-scope `usize` expression, an integer
/// literal checks it against that value, and `_` skips it.
///
/// The pattern length must equal the tensor rank, otherwise the call does not compile. The
/// tensor is borrowed, not moved, and every expression is evaluated once. Axis checks stay on
/// in release builds; a mismatch panics with the call, the axis, the expected and actual
/// sizes, and the full dims.
///
/// ```
/// use burn_tensor::{Tensor, unpack_shape};
///
/// let device = Default::default();
/// let x = Tensor::<3>::zeros([2, 5, 80], &device);
/// let mask = Tensor::<2>::zeros([2, 5], &device);
///
/// let (batch_size, seq_length) = unpack_shape!(x, [B, T, 80]);
/// let (n,) = unpack_shape!(mask, [=batch_size, N]);
/// assert_eq!((batch_size, seq_length, n), (2, 5, 5));
/// ```
///
/// A pattern whose length differs from the rank does not compile:
///
/// ```compile_fail,E0308
/// use burn_tensor::{Tensor, unpack_shape};
/// let x = Tensor::<2>::zeros([2, 3], &Default::default());
/// let (b,) = unpack_shape!(x, [B, 3, 1]);
/// ```
///
/// Neither does a pattern with nothing to return; use [`assert_shape!`](crate::assert_shape)
/// for that:
///
/// ```compile_fail
/// use burn_tensor::{Tensor, unpack_shape};
/// let x = Tensor::<2>::zeros([2, 3], &Default::default());
/// unpack_shape!(x, [2, 3]);
/// ```
#[macro_export]
macro_rules! unpack_shape {
    ($($tt:tt)*) => {
        $crate::__unpack_shape!($crate, $($tt)*)
    };
}

/// Asserts a tensor's rank at compile time and its axis sizes at runtime.
///
/// The pattern has one slot per axis: `=expr` checks the axis against an in-scope `usize`
/// expression, an integer literal checks it against that value, and `_` skips it. Bare
/// identifiers are rejected; use [`unpack_shape!`](crate::unpack_shape) to get axis sizes
/// back.
///
/// The pattern length must equal the tensor rank, otherwise the call does not compile. The
/// tensor is borrowed, not moved, and every expression is evaluated once. Checks stay on in
/// release builds; see [`debug_assert_shape!`](crate::debug_assert_shape) for the debug-only
/// variant.
///
/// ```
/// use burn_tensor::{Tensor, assert_shape};
///
/// let device = Default::default();
/// let batch_size = 2;
/// let y = Tensor::<3>::zeros([batch_size, 5, 256], &device);
/// let mask = Tensor::<2>::zeros([batch_size, 5], &device);
///
/// assert_shape!(y, [=batch_size, _, 256]);
/// assert_shape!(mask, [=batch_size, =y.dims()[1]]);
/// ```
///
/// A mismatch panics with the call and the offending axis:
///
/// ```should_panic
/// use burn_tensor::{Tensor, assert_shape};
/// let x = Tensor::<2>::zeros([2, 3], &Default::default());
/// // assert_shape!(x, [2, 99]): axis 1 expected 99, got 3 (dims [2, 3])
/// assert_shape!(x, [2, 99]);
/// ```
///
/// A pattern whose length differs from the rank does not compile, and neither does a bare
/// identifier:
///
/// ```compile_fail,E0308
/// use burn_tensor::{Tensor, assert_shape};
/// let x = Tensor::<2>::zeros([2, 3], &Default::default());
/// assert_shape!(x, [2, 3, 1]);
/// ```
///
/// ```compile_fail
/// use burn_tensor::{Tensor, assert_shape};
/// let x = Tensor::<2>::zeros([2, 3], &Default::default());
/// assert_shape!(x, [B, 3]);
/// ```
///
/// Only a `Tensor` is accepted. A `Shape` is a type error rather than a weaker check:
///
/// ```compile_fail,E0308
/// use burn_tensor::{Tensor, assert_shape};
/// let x = Tensor::<3>::zeros([2, 3, 4], &Default::default());
/// assert_shape!(x.shape(), [2, 3]);
/// ```
#[macro_export]
macro_rules! assert_shape {
    ($($tt:tt)*) => {
        $crate::__assert_shape!($crate, $($tt)*)
    };
}

/// Same as [`assert_shape!`](crate::assert_shape), but compiled out unless `debug_assertions`
/// is on.
///
/// The rank check is a type check and stays in every build. Like `debug_assert!`, neither the
/// tensor expression nor the `=expr` slots are evaluated in release builds.
///
/// ```
/// use burn_tensor::{Tensor, debug_assert_shape};
///
/// let device = Default::default();
/// let hidden = Tensor::<3>::zeros([2, 5, 256], &device);
/// debug_assert_shape!(hidden, [2, _, 256]);
/// ```
///
/// With debug assertions on, a mismatch panics like [`assert_shape!`](crate::assert_shape):
///
/// ```should_panic
/// use burn_tensor::{Tensor, debug_assert_shape};
/// let x = Tensor::<2>::zeros([2, 3], &Default::default());
/// debug_assert_shape!(x, [2, 99]);
/// # if !cfg!(debug_assertions) { panic!("compiled out") }
/// ```
///
/// A bare identifier does not compile:
///
/// ```compile_fail
/// use burn_tensor::{Tensor, debug_assert_shape};
/// let x = Tensor::<2>::zeros([2, 3], &Default::default());
/// debug_assert_shape!(x, [B, 3]);
/// ```
#[macro_export]
macro_rules! debug_assert_shape {
    ($($tt:tt)*) => {
        $crate::__debug_assert_shape!($crate, $($tt)*)
    };
}
