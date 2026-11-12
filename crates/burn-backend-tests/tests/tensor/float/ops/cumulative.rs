use super::*;
use burn_tensor::TensorData;

use burn_tensor::ElementConversion;

#[test]
fn test_cumsum_float_dim_0() {
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let output = tensor.cumsum(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1.0, 2.0, 3.0], [5.0, 7.0, 9.0]]), false);
}

#[test]
fn test_cumsum_float_dim_1() {
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let output = tensor.cumsum(1);

    output.into_data().assert_eq(
        &TensorData::from([[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]]),
        false,
    );
}

#[test]
fn test_cumsum_float_negative_dim() {
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let output = tensor.cumsum(-1);

    output.into_data().assert_eq(
        &TensorData::from([[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]]),
        false,
    );
}

#[test]
fn test_cumsum_non_contiguous() {
    let tensor = TestTensor::<2>::from([[1., 2.], [3., 4.]]).swap_dims(0, 1);

    let output = tensor.cumsum(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1., 4.], [2., 6.]]), false);
}

#[test]
fn test_cumsum_float_3d() {
    let tensor = TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);

    let output = tensor.cumsum(2);

    output.into_data().assert_eq(
        &TensorData::from([[[1.0, 3.0], [3.0, 7.0]], [[5.0, 11.0], [7.0, 15.0]]]),
        false,
    );
}

#[test]
fn test_cumprod_float_dim_0() {
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let output = tensor.cumprod(0);

    output.into_data().assert_eq(
        &TensorData::from([[1.0, 2.0, 3.0], [4.0, 10.0, 18.0]]),
        false,
    );
}

#[test]
fn test_cumprod_float_dim_1() {
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let output = tensor.cumprod(1);

    output.into_data().assert_eq(
        &TensorData::from([[1.0, 2.0, 6.0], [4.0, 20.0, 120.0]]),
        false,
    );
}

#[test]
fn test_cumprod_float_negative_dim() {
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let output = tensor.cumprod(-1);

    output.into_data().assert_eq(
        &TensorData::from([[1.0, 2.0, 6.0], [4.0, 20.0, 120.0]]),
        false,
    );
}

#[test]
fn test_cumprod_float_3d() {
    let tensor = TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);

    let output = tensor.cumprod(2);

    output.into_data().assert_eq(
        &TensorData::from([[[1.0, 2.0], [3.0, 12.0]], [[5.0, 30.0], [7.0, 56.0]]]),
        false,
    );
}

#[test]
fn test_cummin_float_dim_0() {
    let tensor = TestTensor::<2>::from([[3.0, 1.0, 4.0], [2.0, 5.0, 1.0]]);

    let output = tensor.cummin(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3.0, 1.0, 4.0], [2.0, 1.0, 1.0]]), false);
}

#[test]
fn test_cummin_float_dim_1() {
    let tensor = TestTensor::<2>::from([[3.0, 1.0, 4.0], [2.0, 5.0, 1.0]]);

    let output = tensor.cummin(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3.0, 1.0, 1.0], [2.0, 2.0, 1.0]]), false);
}

#[test]
fn test_cummin_float_negative_dim() {
    let tensor = TestTensor::<2>::from([[3.0, 1.0, 4.0], [2.0, 5.0, 1.0]]);

    let output = tensor.cummin(-1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3.0, 1.0, 1.0], [2.0, 2.0, 1.0]]), false);
}

#[test]
fn test_cummin_float_3d() {
    let tensor = TestTensor::<3>::from([[[4.0, 2.0], [3.0, 1.0]], [[5.0, 6.0], [7.0, 8.0]]]);

    let output = tensor.cummin(2);

    output.into_data().assert_eq(
        &TensorData::from([[[4.0, 2.0], [3.0, 1.0]], [[5.0, 5.0], [7.0, 7.0]]]),
        false,
    );
}

#[test]
fn test_cummax_float_dim_0() {
    let tensor = TestTensor::<2>::from([[3.0, 1.0, 4.0], [1.0, 5.0, 2.0]]);

    let output = tensor.cummax(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3.0, 1.0, 4.0], [3.0, 5.0, 4.0]]), false);
}

#[test]
fn test_cummax_float_dim_1() {
    let tensor = TestTensor::<2>::from([[3.0, 1.0, 4.0], [1.0, 5.0, 2.0]]);

    let output = tensor.cummax(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3.0, 3.0, 4.0], [1.0, 5.0, 5.0]]), false);
}

#[test]
fn test_cummax_float_negative_dim() {
    let tensor = TestTensor::<2>::from([[3.0, 1.0, 4.0], [1.0, 5.0, 2.0]]);

    let output = tensor.cummax(-1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3.0, 3.0, 4.0], [1.0, 5.0, 5.0]]), false);
}

#[test]
fn test_cummax_float_3d() {
    let tensor = TestTensor::<3>::from([[[1.0, 3.0], [2.0, 4.0]], [[5.0, 2.0], [6.0, 1.0]]]);

    let output = tensor.cummax(2);

    output.into_data().assert_eq(
        &TensorData::from([[[1.0, 3.0], [2.0, 4.0]], [[5.0, 5.0], [6.0, 6.0]]]),
        false,
    );
}

// All burn backends should propagate NaN from cummin/cummax. See issue #4814.
#[test]
fn test_cummin_nan_propagation() {
    // Once NaN appears, cummin propagates it forward.
    let tensor = TestTensor::<1>::from([3.0, f32::NAN, 1.0, 2.0]);

    let output = tensor.cummin(0);

    let data: Vec<FloatElem> = output.try_into_vec_as().unwrap();
    assert_eq!(data[0], 3.0.elem::<FloatElem>());
    assert!(data[1].is_nan());
    assert!(data[2].is_nan());
    assert!(data[3].is_nan());
}

#[test]
fn test_cummax_nan_propagation() {
    let tensor = TestTensor::<1>::from([1.0, f32::NAN, 5.0, 2.0]);

    let output = tensor.cummax(0);

    let data: Vec<FloatElem> = output.try_into_vec_as().unwrap();
    assert_eq!(data[0], 1.0.elem::<FloatElem>());
    assert!(data[1].is_nan());
    assert!(data[2].is_nan());
    assert!(data[3].is_nan());
}

#[test]
fn test_cummin_and_cummax_nan_at_start() {
    // NaN on the first element should poison the entire output.
    let tensor = TestTensor::<1>::from([f32::NAN, 1.0, 2.0]);

    for output in [tensor.clone().cummin(0), tensor.cummax(0)] {
        let data: Vec<FloatElem> = output.try_into_vec_as().unwrap();
        assert!(data[0].is_nan());
        assert!(data[1].is_nan());
        assert!(data[2].is_nan());
    }
}

// A half-precision scan summed in its own type stops growing once `sum + 1` rounds back to `sum`:
// at 2048 for f16 and 256 for bf16. Both totals below are exact in their type.
//
// A device without f16 or bf16 skips that dtype, as the other half-precision tests do. Flex
// supports both, so the test checks both there.
#[test]
fn should_cumsum_half_precision_without_stalling() {
    let device = burn_tensor::Device::default();
    for (dtype, n) in [
        (burn_tensor::DType::F16, 4096),
        (burn_tensor::DType::BF16, 512),
    ] {
        if !device.supports_dtype(dtype) {
            continue;
        }
        let ones = TestTensor::<1>::ones([n], &device).cast(dtype);

        let output = ones.cumsum(0).into_data().convert::<f32>();
        let values = output.as_slice::<f32>().unwrap();

        assert_eq!(values[n - 1], n as f32, "{dtype:?}");
        assert_eq!(values[n / 2 - 1], (n / 2) as f32, "{dtype:?}");
    }
}

// A product of `1 + e`, the smallest step above 1 in the type, multiplied in its own type loses
// the `e²` of every step and drifts away from `(1 + e)^n`. Accumulated in f32 and rounded once,
// the result is the half value nearest to `(1 + e)^n`: about 2.7170 for f16 (n = 1024) and 2.7077
// for bf16 (n = 128), neither near a rounding tie.
#[test]
fn should_cumprod_half_precision_without_drifting() {
    let device = burn_tensor::Device::default();
    for (dtype, step, n) in [
        (burn_tensor::DType::F16, 2f64.powi(-10), 1024),
        (burn_tensor::DType::BF16, 2f64.powi(-7), 128),
    ] {
        if !device.supports_dtype(dtype) {
            continue;
        }
        let factors = TestTensor::<1>::full([n], 1.0 + step, &device).cast(dtype);

        let output = factors.cumprod(0).into_data().convert::<f32>();
        let values = output.as_slice::<f32>().unwrap();

        let round = |x: f64| match dtype {
            burn_tensor::DType::F16 => burn_tensor::f16::from_f64(x).to_f32(),
            _ => burn_tensor::bf16::from_f64(x).to_f32(),
        };
        let exact = |count: i32| (1.0 + step).powi(count);
        assert_eq!(values[n - 1], round(exact(n as i32)), "{dtype:?}");
        assert_eq!(values[n / 2 - 1], round(exact(n as i32 / 2)), "{dtype:?}");
    }
}
