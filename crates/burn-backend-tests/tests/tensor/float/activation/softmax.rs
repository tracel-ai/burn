use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

fn assert_many_small_contributions(dtype: burn_tensor::DType, transposed: bool) {
    // Independent 80-digit decimal references for 1 / (1 + 32 * exp(-gap)).
    let (gap, largest, tail, relative_error) = match dtype {
        burn_tensor::DType::F32 => (18.0, 0.9999995126408857, 1.522997232224319e-8, 1e-5),
        // exp(log_softmax) rounds the non-max log probability at magnitude 9 in f16.
        burn_tensor::DType::F16 => (9.0, 0.9960664204223475, 1.229243618016421e-4, 1e-2),
        burn_tensor::DType::BF16 => (8.0, 0.9893792080282476, 3.318997491172619e-4, 1e-3),
        burn_tensor::DType::F64 => (40.0, 0.9999999999999999, 4.248354255291588e-18, 1e-12),
        _ => unreachable!(),
    };
    // 100 rows cross the Rayon chunk boundary. Length 33 exercises SIMD and its scalar tail.
    let shape = [100, 33];
    let mut input = vec![-gap; 100 * 33];
    let mut expected = vec![tail; 100 * 33];
    for row in 0..100 {
        let index = row * 33 + row % 33;
        input[index] = 0.0;
        expected[index] = largest;
    }
    let tensor =
        TestTensor::<2>::from_data(TensorData::new(input, shape), &Default::default()).cast(dtype);
    let output = if transposed {
        activation::softmax(tensor.transpose(), 0).transpose()
    } else {
        activation::softmax(tensor, 1)
    }
    .into_data()
    .convert::<f64>();
    let values: Vec<f64> = output.iter::<f64>().collect();
    for row in 0..100 {
        let dominant = values[row * 33 + row % 33];
        assert!(
            dominant < 1.0,
            "{dtype:?}, row {row}: small contributions were lost: {dominant}"
        );
    }
    output.assert_approx_eq::<f64>(
        &TensorData::new(expected, shape)
            .convert_dtype(dtype)
            .convert::<f64>(),
        Tolerance::relative(relative_error),
    );
}

#[test]
fn test_softmax_many_small_contributions() {
    assert_many_small_contributions(<FloatElem as burn_tensor::Element>::dtype(), false);
}

#[test]
fn test_softmax_many_small_contributions_non_last_axis() {
    assert_many_small_contributions(<FloatElem as burn_tensor::Element>::dtype(), true);
}

#[cfg(feature = "flex")]
#[test]
fn test_softmax_many_small_contributions_f64() {
    assert_many_small_contributions(burn_tensor::DType::F64, false);
}

#[cfg(feature = "flex")]
#[test]
fn test_softmax_many_small_contributions_bf16() {
    assert_many_small_contributions(burn_tensor::DType::BF16, false);
}

#[test]
fn test_softmax_single_finite_class() {
    let tensor = TestTensor::<2>::from([[9.0], [-3.0]]);
    activation::softmax(tensor, 1)
        .into_data()
        .assert_eq(&TensorData::from([[1.0], [1.0]]), false);
}

#[test]
fn test_softmax_invalid_single_class() {
    let tensor = TestTensor::<2>::from([[f32::NEG_INFINITY], [f32::NAN], [f32::INFINITY]]);
    let output = activation::softmax(tensor, 1).into_data();
    assert!(output.iter::<f64>().all(f64::is_nan), "{output:?}");
}

#[test]
fn test_softmax_invalid_multi_class() {
    let tensor = TestTensor::<2>::from([
        [f32::NEG_INFINITY, f32::NEG_INFINITY],
        [f32::NAN, 0.0],
        [f32::INFINITY, 0.0],
    ]);
    let output = activation::softmax(tensor, 1).into_data();
    assert!(output.iter::<f64>().all(f64::is_nan), "{output:?}");
}

#[test]
fn test_softmax_d2() {
    let tensor = TestTensor::<2>::from([[1.0, 7.0], [13.0, -3.0]]);

    let output = activation::softmax(tensor, 1);
    let expected = TensorData::from([[2.472623e-03, 9.975274e-01], [1.0, 1.125352e-07]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_softmax_negative_dim() {
    let tensor = TestTensor::<2>::from([[1.0, 7.0], [13.0, -3.0]]);

    let output = activation::softmax(tensor, -1);
    let expected = TensorData::from([[2.472623e-03, 9.975274e-01], [1.0, 1.125352e-07]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_softmax_d1() {
    let tensor = TestTensor::<1>::from([1.0, 2.0, 3.0]);

    let output = activation::softmax(tensor, 0);

    output.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([0.09003, 0.24473, 0.66524]),
        Tolerance::default().set_half_precision_absolute(2e-3),
    );
}

#[test]
fn test_softmax_d2_varied() {
    let tensor = TestTensor::<2>::from([[-1.0, 0.0, 1.0, 2.0], [0.5, 0.5, 0.5, 0.5]]);

    let output = activation::softmax(tensor, 1);

    output.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([
            [0.03205860, 0.08714432, 0.23688284, 0.64391422],
            [0.25, 0.25, 0.25, 0.25],
        ]),
        Tolerance::default().set_half_precision_absolute(2e-3),
    );
}

#[test]
fn test_softmax_d3_last_axis() {
    let tensor = TestTensor::<3>::from([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
    ]);

    let output = activation::softmax(tensor, 2);

    output.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([
            [
                [0.09003057, 0.24472848, 0.66524094],
                [0.09003057, 0.24472848, 0.66524094],
            ],
            [
                [0.21194156, 0.21194156, 0.57611688],
                [0.33333334, 0.33333334, 0.33333334],
            ],
        ]),
        Tolerance::default().set_half_precision_absolute(2e-3),
    );
}

#[test]
fn test_softmax_non_contiguous_input() {
    // Softmax on a transposed (non-contiguous) input. The [3, 4] tensor is
    // transposed to [4, 3] before softmax over the last axis, exercising
    // stride-aware handling of the op in every backend.
    //
    // Every transposed row has the pattern [a, a+4, a+8] (columns of the
    // original tensor); softmax is shift-invariant, so every row shares
    // the same result.
    let t = TestTensor::<2>::from([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ]);
    let t_transposed = t.transpose();

    let output = activation::softmax(t_transposed, 1);

    let row = [3.2932044e-4, 1.7980287e-2, 0.98169035];
    let expected = TensorData::from([row, row, row, row]);
    output.into_data().assert_approx_eq::<FloatElem>(
        &expected,
        Tolerance::default().set_half_precision_absolute(2e-3),
    );
}

#[test]
fn test_softmax_non_last_axis_d3() {
    let tensor = TestTensor::<3>::from([
        [[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]],
        [[0.1, 2.5, -0.3], [1.2, -0.7, 2.1]],
    ]);

    let output = activation::softmax(tensor.clone(), 1);

    // Reference: softmax along middle axis computed via exp / sum pattern.
    let exp = tensor.exp();
    let sum = exp.clone().sum_dim(1);
    let expected = exp / sum;

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::default());
}

#[test]
fn test_softmax_non_last_axis_d4_dim0() {
    let tensor = TestTensor::<4>::from([
        [[[1.0, -0.5], [0.3, 2.1]], [[-1.2, 0.0], [0.8, 1.5]]],
        [[[0.4, -1.1], [2.0, 0.2]], [[-0.3, 0.9], [1.1, -0.7]]],
    ]);

    let output = activation::softmax(tensor.clone(), 0);

    let exp = tensor.exp();
    let sum = exp.clone().sum_dim(0);
    let expected = exp / sum;

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::default());
}
