use super::*;
use burn_tensor::TensorData;

#[test]
fn multinomial_output_shape() {
    let probs = TestTensor::<2>::from([[0.5, 0.5], [0.5, 0.5]]);
    let samples = probs.multinomial(5, true);

    assert_eq!(samples.dims(), [2, 5]);
}

#[test]
fn multinomial_deterministic_single_category() {
    let probs = TestTensor::<2>::from([[1.0, 0.0, 0.0]]);
    let samples = probs.multinomial(10, true);

    let data = samples.into_data();
    let expected = TensorData::from([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0i64]]);
    data.assert_eq(&expected, false);
}

#[test]
fn multinomial_deterministic_last_category() {
    let probs = TestTensor::<2>::from([[0.0, 0.0, 1.0]]);
    let samples = probs.multinomial(10, true);

    let data = samples.into_data();
    let expected = TensorData::from([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2i64]]);
    data.assert_eq(&expected, false);
}

#[test]
fn multinomial_indices_in_range() {
    let num_categories = 5;
    let probs = TestTensor::<2>::from([[0.2, 0.2, 0.2, 0.2, 0.2]]);
    let samples = probs.multinomial(100, true);

    let data: TensorData = samples.into_data();
    let values = data.to_vec::<i64>().unwrap();
    for &v in &values {
        assert!(v >= 0 && v < num_categories, "index {v} out of range");
    }
}

#[test]
fn multinomial_batch_deterministic() {
    // Row 0: always index 0, Row 1: always index 2
    let probs = TestTensor::<2>::from([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    let samples = probs.multinomial(5, true);

    let data = samples.into_data();
    let expected = TensorData::from([[0, 0, 0, 0, 0i64], [2, 2, 2, 2, 2]]);
    data.assert_eq(&expected, false);
}

#[test]
fn multinomial_single_category_always_zero() {
    let probs = TestTensor::<2>::from([[1.0]]);
    let samples = probs.multinomial(5, true);

    let data = samples.into_data();
    let expected = TensorData::from([[0, 0, 0, 0, 0i64]]);
    data.assert_eq(&expected, false);
}

#[test]
fn multinomial_unnormalized_weights() {
    // Weights [0, 0, 10] — should always sample index 2 after normalization
    let probs = TestTensor::<2>::from([[0.0, 0.0, 10.0]]);
    let samples = probs.multinomial(10, true);

    let data = samples.into_data();
    let expected = TensorData::from([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2i64]]);
    data.assert_eq(&expected, false);
}

#[test]
fn multinomial_statistical_distribution() {
    // With equal probabilities [0.5, 0.5], samples should be roughly evenly split
    let num_samples = 1000;
    let probs = TestTensor::<2>::from([[0.5, 0.5]]);
    let samples = probs.multinomial(num_samples, true);

    let data: TensorData = samples.into_data();
    let values = data.to_vec::<i64>().unwrap();

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
fn multinomial_no_replacement_multiple_samples_panics() {
    let probs = TestTensor::<2>::from([[0.5, 0.5]]);
    let _ = probs.multinomial(2, false);
}
