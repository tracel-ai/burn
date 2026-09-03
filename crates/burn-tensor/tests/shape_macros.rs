//! Integration tests for `unpack_shape!`, `assert_shape!` and `debug_assert_shape!`.
//!
//! Rank mismatches and misuse of the slot kinds are compile errors, covered by the
//! `compile_fail` doctests in `burn_tensor::shape_macro_compile_fail`. The tests here cover
//! the runtime behavior: bindings, axis checks, panic messages, and the debug-only gating of
//! `debug_assert_shape!`.

use burn_tensor::{Int, Tensor, assert_shape, debug_assert_shape, unpack_shape};

fn t<const D: usize>(shape: [usize; D]) -> Tensor<D> {
    Tensor::zeros(shape, &Default::default())
}

struct Config {
    d_model: usize,
}

// ---- unpack_shape!: bindings ----

#[test]
fn unpack_all_fresh_returns_each_dim() {
    let x = t([2, 3, 4]);
    let (b, tlen, c) = unpack_shape!(x, [B, T, C]);
    assert_eq!((b, tlen, c), (2, 3, 4));
}

#[test]
fn unpack_single_fresh_is_a_one_element_tuple() {
    let x = t([5]);
    let (n,) = unpack_shape!(x, [N]);
    assert_eq!(n, 5);
}

#[test]
fn unpack_fresh_plus_literal() {
    let x = t([2, 3, 80]);
    let (b, tlen) = unpack_shape!(x, [B, T, 80]);
    assert_eq!((b, tlen), (2, 3));
}

#[test]
fn unpack_fresh_plus_scope_check() {
    let x = t([2, 3, 4]);
    let b = 2usize;
    let c = 4usize;
    let (tlen,) = unpack_shape!(x, [=b, T, =c]);
    assert_eq!(tlen, 3);
}

#[test]
fn unpack_check_accepts_expressions() {
    let x = t([2, 6, 128]);
    let config = Config { d_model: 128 };
    let n = 3usize;
    let (b,) = unpack_shape!(x, [B, =n * 2, =config.d_model]);
    assert_eq!(b, 2);
}

#[test]
fn unpack_wildcard_skips_axis() {
    let x = t([2, 3, 4]);
    let (b, c) = unpack_shape!(x, [B, _, C]);
    assert_eq!((b, c), (2, 4));
}

#[test]
fn unpack_does_not_move_the_tensor() {
    let x = t([2, 3]);
    let (b, _) = unpack_shape!(x, [B, T]);
    let (b_again, _) = unpack_shape!(&x, [B, T]);
    assert_eq!(b, b_again);
    assert_eq!(x.dims(), [2, 3]);
}

#[test]
fn unpack_works_on_int_tensors() {
    let x: Tensor<2, Int> = Tensor::zeros([4, 7], &Default::default());
    let (rows, cols) = unpack_shape!(x, [R, C]);
    assert_eq!((rows, cols), (4, 7));
}

// ---- unpack_shape!: mismatch panics ----

#[test]
#[should_panic(
    expected = "unpack_shape!(x, [B, T, 80]): axis 2 expected 80, got 4 (dims [2, 3, 4])"
)]
fn unpack_literal_mismatch_panics() {
    let x = t([2, 3, 4]);
    let _ = unpack_shape!(x, [B, T, 80]);
}

#[test]
#[should_panic(expected = "unpack_shape!(x, [=b, T, C]): axis 0 expected 99, got 2")]
fn unpack_scope_mismatch_panics() {
    let x = t([2, 3, 4]);
    let b = 99usize;
    let _ = unpack_shape!(x, [=b, T, C]);
}

#[test]
#[should_panic(
    expected = "unpack_shape!(x, [B, =config.d_model]): axis 1 expected 128, got 6 (dims [2, 6])"
)]
fn unpack_expression_mismatch_panics() {
    let x = t([2, 6]);
    let config = Config { d_model: 128 };
    let _ = unpack_shape!(x, [B, =config.d_model]);
}

// ---- assert_shape!: happy paths ----

#[test]
fn assert_all_literals_match() {
    let x = t([2, 3, 80]);
    assert_shape!(x, [2, 3, 80]);
}

#[test]
fn assert_all_scope_idents_match() {
    let x = t([2, 3, 4]);
    let b = 2usize;
    let tlen = 3usize;
    let c = 4usize;
    assert_shape!(x, [=b, =tlen, =c]);
}

#[test]
fn assert_mixed_slots() {
    let x = t([2, 3, 128]);
    let b = 2usize;
    let config = Config { d_model: 128 };
    assert_shape!(x, [=b, _, =config.d_model]);
}

#[test]
fn assert_only_wildcards_checks_rank() {
    let x = t([2, 3]);
    assert_shape!(x, [_, _]);
}

#[test]
fn assert_from_a_reference() {
    fn check(x: &Tensor<2>) {
        assert_shape!(x, [2, 3]);
    }
    check(&t([2, 3]));
}

// ---- assert_shape!: mismatch panics ----

#[test]
#[should_panic(
    expected = "assert_shape!(x, [2, 99, 4]): axis 1 expected 99, got 3 (dims [2, 3, 4])"
)]
fn assert_literal_mismatch_panics() {
    let x = t([2, 3, 4]);
    assert_shape!(x, [2, 99, 4]);
}

#[test]
#[should_panic(expected = "assert_shape!(x, [=bogus, 3, 4]): axis 0 expected 99, got 2")]
fn assert_scope_mismatch_panics() {
    let x = t([2, 3, 4]);
    let bogus = 99usize;
    assert_shape!(x, [=bogus, 3, 4]);
}

// ---- debug_assert_shape! ----

#[test]
fn debug_assert_matching_shape_passes() {
    let x = t([2, 3]);
    let b = 2usize;
    debug_assert_shape!(x, [=b, 3]);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "debug_assert_shape!(x, [2, 99]): axis 1 expected 99, got 3")]
fn debug_assert_mismatch_panics_in_debug() {
    let x = t([2, 3]);
    debug_assert_shape!(x, [2, 99]);
}

#[test]
#[cfg(not(debug_assertions))]
fn debug_assert_mismatch_compiles_out_in_release() {
    let x = t([2, 3]);
    debug_assert_shape!(x, [99, 99]);
}
