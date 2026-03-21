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
#[should_panic]
fn multinomial_no_replacement_multiple_samples_panics() {
    let probs = TestTensor::<2>::from([[0.5, 0.5]]);
    let _ = probs.multinomial(2, false);
}
