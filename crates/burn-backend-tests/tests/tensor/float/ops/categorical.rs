use super::*;
use burn_tensor::TensorData;

#[test]
fn categorical_output_shape() {
    let probs = TestTensor::<2>::from([[0.5, 0.5], [0.5, 0.5]]);
    let samples = probs.categorical(5);

    assert_eq!(samples.dims(), [2, 5]);
}

#[test]
fn categorical_1d() {
    // 1D input: single distribution
    let probs = TestTensor::<1>::from([0.0, 0.0, 1.0]);
    let samples = probs.categorical(5);

    assert_eq!(samples.dims(), [5]);
    let data = samples.into_data();
    let expected = TensorData::from([2, 2, 2, 2, 2i64]);
    data.assert_eq(&expected, false);
}

#[test]
fn categorical_3d() {
    // 3D input: [2, 1, 3] — two batches, each with one sub-batch of 3 categories
    let device = Default::default();
    let probs = TestTensor::<3>::from_data(
        TensorData::from([[[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]]),
        &device,
    );
    let samples = probs.categorical(4);

    assert_eq!(samples.dims(), [2, 1, 4]);
    let data = samples.into_data();
    let expected = TensorData::from([[[0, 0, 0, 0i64]], [[2, 2, 2, 2]]]);
    data.assert_eq(&expected, false);
}

#[test]
fn categorical_deterministic_single_category() {
    let probs = TestTensor::<2>::from([[1.0, 0.0, 0.0]]);
    let samples = probs.categorical(10);

    let data = samples.into_data();
    let expected = TensorData::from([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0i64]]);
    data.assert_eq(&expected, false);
}

#[test]
fn categorical_deterministic_last_category() {
    let probs = TestTensor::<2>::from([[0.0, 0.0, 1.0]]);
    let samples = probs.categorical(10);

    let data = samples.into_data();
    let expected = TensorData::from([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2i64]]);
    data.assert_eq(&expected, false);
}

#[test]
fn categorical_indices_in_range() {
    let num_categories = 5;
    let probs = TestTensor::<2>::from([[0.2, 0.2, 0.2, 0.2, 0.2]]);
    let samples = probs.categorical(100);

    let data = samples.into_data();
    data.assert_within_range::<IntElem>(0..num_categories);
}

#[test]
fn categorical_batch_deterministic() {
    // Row 0: always index 0, Row 1: always index 2
    let probs = TestTensor::<2>::from([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    let samples = probs.categorical(5);

    let data = samples.into_data();
    let expected = TensorData::from([[0, 0, 0, 0, 0i64], [2, 2, 2, 2, 2]]);
    data.assert_eq(&expected, false);
}

#[test]
fn categorical_single_category_always_zero() {
    let probs = TestTensor::<2>::from([[1.0]]);
    let samples = probs.categorical(5);

    let data = samples.into_data();
    let expected = TensorData::from([[0, 0, 0, 0, 0i64]]);
    data.assert_eq(&expected, false);
}

#[test]
fn categorical_unnormalized_weights() {
    // Weights [0, 0, 10] — should always sample index 2 after normalization
    let probs = TestTensor::<2>::from([[0.0, 0.0, 10.0]]);
    let samples = probs.categorical(10);

    let data = samples.into_data();
    let expected = TensorData::from([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2i64]]);
    data.assert_eq(&expected, false);
}

#[test]
fn categorical_statistical_distribution() {
    // With equal probabilities [0.5, 0.5], samples should be roughly evenly split
    let num_samples = 1000;
    let probs = TestTensor::<2>::from([[0.5, 0.5]]);
    let samples = probs.categorical(num_samples);

    let data: TensorData = samples.into_data();
    let values = data.to_vec::<IntElem>().unwrap();

    let count_zero = values.iter().filter(|&&v| v == 0).count();
    let ratio = count_zero as f64 / num_samples as f64;

    // Expect ~50% zeros, allow wide tolerance for stochastic test
    assert!(
        ratio > 0.3 && ratio < 0.7,
        "expected ~50% index-0 samples, got {:.1}%",
        ratio * 100.0
    );
}

#[test]
#[should_panic]
fn categorical_zero_samples_panics() {
    let probs = TestTensor::<2>::from([[0.5, 0.5]]);
    let _ = probs.categorical(0);
}
