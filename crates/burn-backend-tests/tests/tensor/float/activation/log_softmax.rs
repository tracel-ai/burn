use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_log_softmax_d2() {
    let tensor = TestTensor::<2>::from([[1.0, 0.0], [0.0, 1.0]]);

    let output = activation::log_softmax(tensor, 1);
    let expected = TensorData::from([[-0.3132617, -1.3132617], [-1.3132617, -0.3132617]]);

    let tolerance = Tolerance::rel_abs(0.01, 0.0001);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn test_log_softmax_negative_dim() {
    let tensor = TestTensor::<2>::from([[1.0, 0.0], [0.0, 1.0]]);

    let output = activation::log_softmax(tensor, -1);
    let expected = TensorData::from([[-0.3132617, -1.3132617], [-1.3132617, -0.3132617]]);

    let tolerance = Tolerance::rel_abs(0.01, 0.0001);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

fn assert_high_confidence_log_softmax(dtype: burn_tensor::DType) {
    // Independent f64 references: log(p_max) = -ln_1p(2 * exp(-gap)).
    // The f16 tails stay above its minimum normal value, avoiding device flush-to-zero.
    let (gap, correction, relative_error) = match dtype {
        burn_tensor::DType::F16 => (9.0, 2.467891532250189e-4, 1e-2),
        burn_tensor::DType::BF16 => (8.0, 6.707002860752102e-4, 2e-2),
        burn_tensor::DType::F32 => (18.0, 3.04599590255207e-8, 1e-5),
        burn_tensor::DType::F64 => (40.0, 8.496708510583178e-18, 1e-12),
        _ => unreachable!(),
    };
    let tensor = TestTensor::<2>::from([[gap / 2.0, -gap / 2.0, -gap / 2.0]]).cast(dtype);
    let output = activation::log_softmax(tensor, 1).into_data();
    let dominant = output.iter::<f64>().next().unwrap();

    assert!(
        dominant < 0.0,
        "{dtype:?}: expected a negative log probability, got {dominant}"
    );
    output.convert::<f64>().assert_approx_eq::<f64>(
        &TensorData::from([[-correction, -gap - correction, -gap - correction]]),
        Tolerance::relative(relative_error),
    );
}

#[test]
fn test_log_softmax_high_confidence() {
    assert_high_confidence_log_softmax(<FloatElem as burn_tensor::Element>::dtype());
}

#[cfg(feature = "flex")]
#[test]
fn test_log_softmax_high_confidence_bf16() {
    assert_high_confidence_log_softmax(burn_tensor::DType::BF16);
}

#[cfg(feature = "flex")]
#[test]
fn test_log_softmax_high_confidence_f64() {
    assert_high_confidence_log_softmax(burn_tensor::DType::F64);
}

#[test]
fn test_log_softmax_single_finite_class() {
    let tensor = TestTensor::<2>::from([[9.0], [-3.0]]);
    activation::log_softmax(tensor, 1)
        .into_data()
        .assert_eq(&TensorData::from([[0.0], [0.0]]), false);
}

#[test]
fn test_log_softmax_invalid_single_class() {
    let tensor = TestTensor::<2>::from([[f32::NEG_INFINITY], [f32::NAN], [f32::INFINITY]]);
    let output = activation::log_softmax(tensor, 1).into_data();
    assert!(output.iter::<f64>().all(f64::is_nan), "{output:?}");
}

#[test]
fn test_log_softmax_invalid_multi_class() {
    let tensor = TestTensor::<2>::from([
        [f32::NEG_INFINITY, f32::NEG_INFINITY],
        [f32::NAN, 0.0],
        [f32::INFINITY, 0.0],
    ]);
    let output = activation::log_softmax(tensor, 1).into_data();
    assert!(output.iter::<f64>().all(f64::is_nan), "{output:?}");
}

#[test]
fn test_log_softmax_tied_maxima() {
    let tensor = TestTensor::<2>::from([[2.0, 2.0, 0.0], [-2.0, 1.0, 1.0]]);
    let expected = TensorData::from([
        [
            -0.7586236756795135,
            -0.7586236756795135,
            -2.7586236756795133,
        ],
        [
            -3.7177359186667025,
            -0.7177359186667026,
            -0.7177359186667026,
        ],
    ]);
    activation::log_softmax(tensor, 1)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_log_softmax_transposed_non_last_axis() {
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 4.0], [-1.0, 3.0, 0.0]]).transpose();
    let expected = TensorData::from([
        [-3.1698460195562856, -4.065883903757429],
        [-2.1698460195562856, -0.06588390375742911],
        [-0.16984601955628567, -3.0658839037574293],
    ]);
    activation::log_softmax(tensor, 0)
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_log_softmax_masked_classes() {
    let tensor = TestTensor::<2>::from([
        [0.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
        [f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY],
    ]);
    let expected = TensorData::from([
        [0.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
        [f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY],
    ]);
    activation::log_softmax(tensor, 1)
        .into_data()
        .assert_eq(&expected, false);
}

#[cfg(feature = "flex")]
#[test]
#[should_panic(expected = "max_dim: dimension 1 has size 0")]
fn test_log_softmax_empty_class_dimension_is_rejected() {
    let tensor = TestTensor::<2>::empty([2, 0], &Default::default());
    let _ = activation::log_softmax(tensor, 1);
}

#[test]
fn test_log_softmax_mixed_valid_and_fully_masked_rows() {
    let tensor = TestTensor::<2>::from([
        [0.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
        [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
        [0.0, 1.0, 0.0],
    ]);
    let output = activation::log_softmax(tensor, 1).into_data();
    let values: Vec<f64> = output.iter::<f64>().collect();
    assert_eq!(&values[..3], &[0.0, f64::NEG_INFINITY, f64::NEG_INFINITY]);
    assert!(values[3..6].iter().all(|value| value.is_nan()));
    TensorData::new(values[6..].to_vec(), [3]).assert_approx_eq::<FloatElem>(
        &TensorData::from([-1.551444713932051, -0.5514447139320511, -1.551444713932051]),
        Tolerance::default(),
    );
}
