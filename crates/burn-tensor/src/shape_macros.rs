//! Shape assertion macros. Parsing and expansion live in burn-derive; these wrappers forward
//! `$crate` so the expansion can name [`Tensor::dims`](crate::Tensor::dims) by path, which
//! makes anything other than a `Tensor` a type error.
//!
//! Inspired by the `burn-contracts` crate by Crutcher Dunnavant.

/// Asserts a tensor's rank at compile time and its axis sizes at runtime.
///
/// The pattern has one slot per axis. A slot is either `_`, which skips the axis, or any `usize`
/// expression the axis must equal: a name from `dims()`, a config field, a literal, arithmetic
/// on any of those. To name axes, destructure `tensor.dims()` first and use the names as slots.
///
/// The pattern length must equal the tensor rank, otherwise the call does not compile. One
/// slot may be `..`, which stands for any number of axes; the pattern then fixes only a
/// minimum rank, checked at runtime, and fits a function generic over `const D: usize`. The
/// tensor is borrowed, not moved, and every expression is evaluated once. Checks stay on in
/// release builds; see [`debug_assert_shape!`](crate::debug_assert_shape) for the debug-only
/// variant.
///
/// ```
/// use burn_tensor::{Tensor, assert_shape};
///
/// let device = Default::default();
/// let x = Tensor::<3>::zeros([2, 5, 256], &device);
/// let mask = Tensor::<2>::zeros([2, 5], &device);
/// let patch = Tensor::<2>::zeros([6, 2], &device);
///
/// let [batch_size, seq_length, _] = x.dims();
/// let flat = x.clone().reshape([batch_size * seq_length, 256]);
///
/// assert_shape!(x, [batch_size, seq_length, 256]);     // names from dims() and a literal
/// assert_shape!(flat, [batch_size * seq_length, 256]); // arithmetic on names
/// assert_shape!(patch, [3 * 2, 2]);                    // arithmetic on literals
/// assert_shape!(mask, [batch_size, _]);                // skip an axis
/// ```
///
/// With `..`, the same check works whatever the rank:
///
/// ```
/// use burn_tensor::{Tensor, assert_shape};
///
/// fn check_features<const D: usize>(x: &Tensor<D>, d_model: usize) {
///     assert_shape!(x, [.., d_model]);
/// }
///
/// fn check_channels<const D: usize>(x: &Tensor<D>, channels: usize) {
///     assert_shape!(x, [_, channels, ..]);
/// }
///
/// let device = Default::default();
/// check_features(&Tensor::<2>::zeros([5, 256], &device), 256);
/// check_features(&Tensor::<4>::zeros([2, 3, 5, 256], &device), 256);
/// check_channels(&Tensor::<3>::zeros([2, 16, 100], &device), 16);
/// check_channels(&Tensor::<5>::zeros([2, 16, 4, 8, 8], &device), 16);
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
/// A pattern whose length differs from the rank does not compile:
///
/// ```compile_fail,E0308
/// use burn_tensor::{Tensor, assert_shape};
/// let x = Tensor::<2>::zeros([2, 3], &Default::default());
/// assert_shape!(x, [2, 3, 1]);
/// ```
///
/// Neither does a slot that is not a `usize`:
///
/// ```compile_fail,E0308
/// use burn_tensor::{Tensor, assert_shape};
/// let x = Tensor::<2>::zeros([2, 3], &Default::default());
/// assert_shape!(x, [2i32, 3]);
/// ```
///
/// Only a `Tensor` is accepted. A `Shape` is a type error rather than a weaker check:
///
/// ```compile_fail,E0308
/// use burn_tensor::{Tensor, assert_shape};
/// let x = Tensor::<3>::zeros([2, 3, 4], &Default::default());
/// assert_shape!(x.shape(), [2, 3]);
/// ```
///
/// A pattern may contain `..` only once:
///
/// ```compile_fail
/// use burn_tensor::{Tensor, assert_shape};
/// let x = Tensor::<3>::zeros([2, 3, 4], &Default::default());
/// assert_shape!(x, [.., 3, ..]);
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
/// tensor expression nor the slot expressions are evaluated in release builds.
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
#[macro_export]
macro_rules! debug_assert_shape {
    ($($tt:tt)*) => {
        $crate::__debug_assert_shape!($crate, $($tt)*)
    };
}
