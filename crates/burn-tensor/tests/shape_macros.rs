//! Integration tests for `assert_shape!` and `debug_assert_shape!`.
//!
//! Rank mismatches, non-tensor arguments and non-`usize` slots are compile errors, covered by
//! the `compile_fail` doctests on the macros themselves. The tests here cover the runtime
//! behavior: axis checks, panic messages, evaluation counts, hygiene, and the debug-only gating
//! of `debug_assert_shape!`.

use burn_tensor::{Int, Tensor, assert_shape, debug_assert_shape};
use core::cell::Cell;

fn t<const D: usize>(shape: [usize; D]) -> Tensor<D> {
    Tensor::zeros(shape, &Default::default())
}

struct Config {
    d_model: usize,
}

// ---- assert_shape!: happy paths ----

#[test]
fn assert_all_literals_match() {
    let x = t([2, 3, 80]);
    assert_shape!(x, [2, 3, 80]);
}

#[test]
fn assert_names_from_dims_match() {
    let x = t([2, 3, 4]);
    let y = t([2, 3, 4]);
    let [b, tlen, c] = x.dims();
    assert_shape!(y, [b, tlen, c]);
}

#[test]
fn assert_accepts_expressions() {
    let x = t([2, 6, 128]);
    let config = Config { d_model: 128 };
    let n = 3usize;
    assert_shape!(x, [2, n * 2, config.d_model]);
}

#[test]
fn assert_wildcard_skips_axis() {
    let x = t([2, 3, 4]);
    let b = 2usize;
    assert_shape!(x, [b, _, 4]);
}

#[test]
fn assert_only_wildcards_compiles_to_a_rank_check() {
    let x = t([2, 3]);
    assert_shape!(x, [_, _]);
}

#[test]
fn assert_does_not_move_the_tensor() {
    let x = t([2, 3]);
    assert_shape!(x, [2, 3]);
    assert_shape!(&x, [2, 3]);
    assert_eq!(x.dims(), [2, 3]);
}

#[test]
fn assert_from_a_reference() {
    fn check(x: &Tensor<2>) {
        assert_shape!(x, [2, 3]);
    }
    check(&t([2, 3]));
}

#[test]
fn assert_works_on_int_tensors() {
    let x: Tensor<2, Int> = Tensor::zeros([4, 7], &Default::default());
    assert_shape!(x, [4, 7]);
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
#[should_panic(expected = "assert_shape!(x, [bogus, 3, 4]): axis 0 expected 99, got 2")]
fn assert_name_mismatch_panics() {
    let x = t([2, 3, 4]);
    let bogus = 99usize;
    assert_shape!(x, [bogus, 3, 4]);
}

#[test]
#[should_panic(
    expected = "assert_shape!(x, [_, config.d_model]): axis 1 expected 128, got 6 (dims [2, 6])"
)]
fn assert_expression_mismatch_panics() {
    let x = t([2, 6]);
    let config = Config { d_model: 128 };
    assert_shape!(x, [_, config.d_model]);
}

#[test]
#[should_panic(expected = "axis 0 expected 99, got 2")]
fn assert_reports_the_first_failing_axis() {
    let x = t([2, 3]);
    assert_shape!(x, [99, 99]);
}

// ---- debug_assert_shape! ----

#[test]
fn debug_assert_matching_shape_passes() {
    let x = t([2, 3]);
    let b = 2usize;
    debug_assert_shape!(x, [b, 3]);
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

#[test]
fn debug_assert_evaluates_arguments_only_in_debug() {
    let x = t([2, 3]);
    let evals = Cell::new(0);
    debug_assert_shape!(
        {
            evals.set(evals.get() + 1);
            &x
        },
        [2, {
            evals.set(evals.get() + 1);
            3
        }]
    );
    assert_eq!(evals.get(), if cfg!(debug_assertions) { 2 } else { 0 });
}

// ---- grammar, evaluation, hygiene ----

#[test]
#[rustfmt::skip]
fn accepts_trailing_comma() {
    let x = t([2, 3]);
    assert_shape!(x, [2, 3,]);
    debug_assert_shape!(x, [2, _,]);
}

#[test]
fn slot_expressions_evaluate_once() {
    let x = t([2, 3]);
    let evals = Cell::new(0);
    assert_shape!(
        x,
        [2, {
            evals.set(evals.get() + 1);
            3
        }]
    );
    assert_eq!(evals.get(), 1);
}

#[test]
fn internals_do_not_shadow_caller_names() {
    let x = t([2, 3]);
    let __tensor = 2usize;
    let __dims = 3usize;
    let __expected = 3usize;
    assert_shape!(x, [__tensor, __dims]);
    debug_assert_shape!(x, [_, __expected]);
}
