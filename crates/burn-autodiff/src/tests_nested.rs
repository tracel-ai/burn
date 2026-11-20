//! Higher-order derivatives through the decomposed attention and CTC paths.
//!
//! The explicit native attention/CTC backwards are first-order only. With nested autodiff the
//! outer level sees `Autodiff<B, C>` as its inner backend, whose `has_attention_backward` and
//! `has_ctc_loss_backward` are the `false` trait defaults, so attention and CTC take the
//! decomposed, autodiff-visible path. These tests pin that path: the first-order gradient stays
//! tracked and the second derivative matches finite differences.

use crate::{
    backend::Autodiff,
    checkpoint::strategy::{BalancedCheckpointing, NoCheckpointing},
    tensor::AutodiffTensor,
};
use alloc::{format, vec::Vec};
use burn_backend::{
    TensorData,
    ops::{AttentionModuleOptions, FloatTensorOps, IntTensorOps, ModuleOps},
};
use burn_flex::Flex;
use burn_std::Shape;

type Inner = Autodiff<Flex, NoCheckpointing>;
type Outer = Autodiff<Inner, NoCheckpointing>;
type InnerBalanced = Autodiff<Flex, BalancedCheckpointing>;
type OuterBalanced = Autodiff<InnerBalanced, BalancedCheckpointing>;

fn value(tensor: burn_flex::FlexTensor) -> f32 {
    all_values(tensor)[0]
}

fn all_values(tensor: burn_flex::FlexTensor) -> Vec<f32> {
    let data = burn_std::future::block_on(Flex::float_into_data(tensor)).expect("data");
    data.as_slice::<f32>().expect("f32").to_vec()
}

fn assert_second_close(numeric: f32, analytic: f32, what: &str) {
    let tol = 1e-3 * numeric.abs().max(analytic.abs()).max(1e-6);
    assert!(
        (numeric - analytic).abs() <= tol,
        "{what}: numeric {numeric} vs analytic {analytic} (tol {tol})"
    );
}

/// Second derivative of the attention output sum, non-checkpointed, as the reference.
fn attention_second_analytic<const D: usize>(
    q_values: &[f32],
    k_values: &[f32],
    v_values: &[f32],
    shape: [usize; D],
) -> Vec<f32> {
    let options = AttentionModuleOptions::default();
    let (qi, q) = leaf_pair(q_values, shape);
    let (_ki, k) = leaf_pair(k_values, shape);
    let (_vi, v) = leaf_pair(v_values, shape);
    let out = Outer::attention(q.clone(), k, v, None, None, None, options);
    let g1 = out.backward();
    let dq = q.grad(&g1).expect("first-order attention gradient");
    assert!(dq.is_tracked(), "first-order attention gradient is tracked");
    let d2 = qi.grad(&dq.clone().backward()).expect("second derivative");
    all_values(d2)
}

fn leaf_inner<const D: usize>(values: &[f32], shape: [usize; D]) -> AutodiffTensor<Flex> {
    let device = Default::default();
    AutodiffTensor::<Flex>::new(Flex::float_from_data(
        TensorData::new(values.to_vec(), Shape::new(shape)),
        &device,
    ))
    .require_grad()
}

fn leaf_pair<const D: usize>(
    values: &[f32],
    shape: [usize; D],
) -> (AutodiffTensor<Flex>, AutodiffTensor<Inner>) {
    let inner = leaf_inner(values, shape);
    let outer = AutodiffTensor::<Inner>::new(inner.clone()).require_grad();
    (inner, outer)
}

/// `sum` of the first-order attention gradient, used for the finite-difference comparison.
fn attention_first_total(
    q_values: &[f32],
    k_values: &[f32],
    v_values: &[f32],
    shape: [usize; 4],
) -> f32 {
    let q = AutodiffTensor::<Inner>::new(leaf_inner(q_values, shape)).require_grad();
    let k = AutodiffTensor::<Inner>::new(leaf_inner(k_values, shape));
    let v = AutodiffTensor::<Inner>::new(leaf_inner(v_values, shape));
    let out = Outer::attention(
        q.clone(),
        k,
        v,
        None,
        None,
        None,
        AttentionModuleOptions::default(),
    );
    let g = out.backward();
    let dq = q.grad(&g).expect("dq");
    all_values(dq.into_primitive()).iter().sum()
}

/// Normalized log-probabilities for a `[t, 1, c]` sequence, so the CTC loss is nonlinear in the leaf.
fn ctc_data(t: usize, c: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(t * c);
    for time in 0..t {
        for class in 0..c {
            // Deterministic, non-normalized log-probabilities: CTC only needs finite log-probs,
            // and this keeps the test no_std (no exp/ln).
            data.push(-0.5 - 0.3 * class as f32 - 0.1 * time as f32);
        }
    }
    data
}

/// `sum(dy * weights)` for the first-order CTC gradient, used for finite differences.
fn ctc_first_weighted(data: &[f32], weights: &[f32], t: usize, c: usize) -> f32 {
    let device = Default::default();
    let inner = AutodiffTensor::<Flex>::new(Flex::float_from_data(
        TensorData::new(data.to_vec(), Shape::new([t, 1, c])),
        &device,
    ))
    .require_grad();
    let log_probs = AutodiffTensor::<Inner>::new(inner).require_grad();
    let targets = Flex::int_from_data(TensorData::from([[1i32]]), &device);
    let input_lengths = Flex::int_from_data(TensorData::from([t as i32]), &device);
    let target_lengths = Flex::int_from_data(TensorData::from([1i32]), &device);
    let loss = Outer::float_sum(Outer::ctc_loss(
        log_probs.clone(),
        targets,
        input_lengths,
        target_lengths,
        0,
    ));
    let g = loss.backward();
    let dl = log_probs.grad(&g).expect("dl");
    let weights = AutodiffTensor::<Flex>::new(Flex::float_from_data(
        TensorData::new(weights.to_vec(), Shape::new([t, 1, c])),
        &device,
    ));
    let scalar = Inner::float_sum(Inner::float_mul(dl, weights));
    value(scalar.into_primitive())
}

/// Second derivative of `sum(dy * weights)` for the nested CTC graph.
fn ctc_second(data: &[f32], weights: &[f32], t: usize, c: usize) -> Vec<f32> {
    let device = Default::default();
    let inner = AutodiffTensor::<Flex>::new(Flex::float_from_data(
        TensorData::new(data.to_vec(), Shape::new([t, 1, c])),
        &device,
    ))
    .require_grad();
    let log_probs = AutodiffTensor::<Inner>::new(inner.clone()).require_grad();
    let targets = Flex::int_from_data(TensorData::from([[1i32]]), &device);
    let input_lengths = Flex::int_from_data(TensorData::from([t as i32]), &device);
    let target_lengths = Flex::int_from_data(TensorData::from([1i32]), &device);
    let loss = Outer::float_sum(Outer::ctc_loss(
        log_probs.clone(),
        targets,
        input_lengths,
        target_lengths,
        0,
    ));
    let g1 = loss.backward();
    let dl = log_probs.grad(&g1).expect("dl");
    assert!(dl.is_tracked(), "first-order CTC gradient is tracked");

    let weights = AutodiffTensor::<Flex>::new(Flex::float_from_data(
        TensorData::new(weights.to_vec(), Shape::new([t, 1, c])),
        &device,
    ));
    let scalar = Inner::float_sum(Inner::float_mul(dl, weights));
    let g2 = scalar.backward();
    let d2 = inner.grad(&g2).expect("second-order CTC gradient");
    all_values(d2)
}

#[test]
fn nested_attention_second_derivative_matches_finite_difference() {
    let shape = [1, 1, 2, 2];
    let q_values = [0.1f32, -0.2, 0.3, 0.4];
    let k_values = [0.2f32, 0.1, -0.3, 0.25];
    let v_values = [0.5f32, -0.1, 0.2, 0.3];
    let options = AttentionModuleOptions::default();

    let analytic = attention_second_analytic(&q_values, &k_values, &v_values, shape);

    let eps = 1e-1;
    for i in 0..q_values.len() {
        let mut plus = q_values;
        plus[i] += eps;
        let mut minus = q_values;
        minus[i] -= eps;
        let numeric = (attention_first_total(&plus, &k_values, &v_values, shape)
            - attention_first_total(&minus, &k_values, &v_values, shape))
            / (2.0 * eps);
        assert_second_close(numeric, analytic[i], &format!("attention d2[{i}]"));
    }
}

#[test]
fn nested_ctc_second_derivative_matches_finite_difference() {
    let t = 3usize;
    let c = 3usize;
    let data = ctc_data(t, c);
    // Non-uniform seed: the sum-direction derivative is structurally zero (see the next test).
    let weights: Vec<f32> = (0..t * c).map(|i| 0.1 + 0.03 * i as f32).collect();
    let analytic = ctc_second(&data, &weights, t, c);

    let eps = 1e-1;
    for i in 0..data.len() {
        let mut plus = data.clone();
        plus[i] += eps;
        let mut minus = data.clone();
        minus[i] -= eps;
        let numeric = (ctc_first_weighted(&plus, &weights, t, c)
            - ctc_first_weighted(&minus, &weights, t, c))
            / (2.0 * eps);
        assert_second_close(numeric, analytic[i], &format!("ctc d2[{i}]"));
    }
    assert!(
        analytic.iter().any(|v| v.abs() > 1e-8),
        "the non-uniform seed must exercise a non-zero second derivative"
    );
}

#[test]
fn nested_ctc_sum_direction_second_derivative_is_zero() {
    let t = 3usize;
    let c = 3usize;
    let data = ctc_data(t, c);
    // A uniform seed sums the gradient entries. Adding a common constant to all class log-probs at
    // one time step shifts the CTC loss by a linear amount, so that sum is constant `-T` and this
    // second directional derivative is structurally zero. Keep it as an explicit expectation.
    let analytic = ctc_second(&data, &alloc::vec![1.0; t * c], t, c);
    assert!(analytic.iter().all(|v| v.abs() < 1e-6), "{analytic:?}");
}

#[test]
fn nested_attention_second_derivative_with_balanced_checkpointing() {
    let shape = [1, 1, 2, 2];
    let device = Default::default();
    let q_values = [0.1f32, -0.2, 0.3, 0.4];
    let k_values = [0.2f32, 0.1, -0.3, 0.25];
    let v_values = [0.5f32, -0.1, 0.2, 0.3];
    let options = AttentionModuleOptions::default();

    let qi = AutodiffTensor::<Flex>::new(Flex::float_from_data(
        TensorData::new(q_values.to_vec(), Shape::new(shape)),
        &device,
    ))
    .require_grad();
    let q = AutodiffTensor::<InnerBalanced>::new(qi.clone()).require_grad();
    let k =
        AutodiffTensor::<InnerBalanced>::new(AutodiffTensor::<Flex>::new(Flex::float_from_data(
            TensorData::new(k_values.to_vec(), Shape::new(shape)),
            &device,
        )));
    let v =
        AutodiffTensor::<InnerBalanced>::new(AutodiffTensor::<Flex>::new(Flex::float_from_data(
            TensorData::new(v_values.to_vec(), Shape::new(shape)),
            &device,
        )));

    let out = OuterBalanced::attention(q.clone(), k, v, None, None, None, options);
    let g1 = out.backward();
    let dq = q.grad(&g1).expect("first-order attention gradient");
    assert!(dq.is_tracked(), "first-order gradient is tracked");
    let d2 = qi.grad(&dq.backward()).expect("second derivative");
    let balanced = all_values(d2);
    let reference = attention_second_analytic(&q_values, &k_values, &v_values, shape);
    for (i, (balanced, reference)) in balanced.iter().zip(reference.iter()).enumerate() {
        assert_second_close(
            *balanced,
            *reference,
            &format!("balanced attention d2[{i}]"),
        );
    }
}
