use super::*;
use burn_tensor::{Element, TensorData, Tolerance, module::group_norm};

#[test]
fn test_group_norm_forward() {
    let input = TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]);
    let gamma = TestTensor::<1>::from([1.0, 2.0, 3.0, 4.0]);
    let beta = TestTensor::<1>::from([0.0, 1.0, 2.0, 3.0]);
    assert_eq!(input.dtype(), FloatElem::dtype());

    let output = group_norm(input, Some(gamma), Some(beta), 2, 0.0);
    assert_eq!(output.dtype(), FloatElem::dtype());

    let expected = TensorData::from([[
        [-1.3416408, -0.4472136],
        [1.8944272, 3.6832817],
        [-2.0249224, 0.6583592],
        [4.7888546, 8.366563],
    ]]);
    output.into_data().assert_approx_eq::<FloatElem>(
        &expected,
        Tolerance::relative(1e-5).set_half_precision_relative(5e-3),
    );
}

#[test]
fn test_group_norm_forward_without_affine() {
    let input = TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]);

    let output = group_norm(input, None, None, 2, 0.0);

    let expected = TensorData::from([[
        [-1.3416408, -0.4472136],
        [0.4472136, 1.3416408],
        [-1.3416408, -0.4472136],
        [0.4472136, 1.3416408],
    ]]);
    output.into_data().assert_approx_eq::<FloatElem>(
        &expected,
        Tolerance::relative(1e-5).set_half_precision_relative(5e-3),
    );
}

#[test]
fn test_group_norm_forward_with_partial_affine() {
    let input = TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]);
    let gamma = TestTensor::<1>::from([2.0, 2.0, 2.0, 2.0]);
    let beta = TestTensor::<1>::from([1.0, 1.0, 1.0, 1.0]);

    let scale_only = group_norm(input.clone(), Some(gamma), None, 2, 0.0);
    let shift_only = group_norm(input, None, Some(beta), 2, 0.0);

    let expected_scale = TensorData::from([[
        [-2.6832816, -0.8944272],
        [0.8944272, 2.6832816],
        [-2.6832816, -0.8944272],
        [0.8944272, 2.6832816],
    ]]);
    let expected_shift = TensorData::from([[
        [-0.3416408, 0.5527864],
        [1.4472136, 2.3416408],
        [-0.3416408, 0.5527864],
        [1.4472136, 2.3416408],
    ]]);
    let tolerance = Tolerance::relative(1e-5).set_half_precision_relative(5e-3);

    scale_only
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_scale, tolerance);
    shift_only
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_shift, tolerance);
}

#[test]
fn test_group_norm_non_contiguous_input() {
    let input =
        TestTensor::<3>::from([[[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]]]).swap_dims(1, 2);
    let contiguous = TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]);

    let output = group_norm(input, None, None, 2, 1e-5);
    let expected = group_norm(contiguous, None, None, 2, 1e-5);

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected.into_data(),
        Tolerance::relative(1e-5).set_half_precision_relative(5e-3),
    );
}

#[test]
fn test_group_norm_channel_innermost_traversal_with_affine() {
    let input = TestTensor::<4>::from(TensorData::new(
        (1..=48).map(|value| value as f32).collect(),
        [2, 2, 3, 4],
    ))
    .permute([0, 3, 1, 2]);
    let contiguous = TestTensor::<4>::from_data(input.to_data(), &input.device());
    let gamma = TestTensor::<1>::from([0.5, 1.0, 1.5, 2.0]);
    let beta = TestTensor::<1>::from([-1.0, 0.0, 1.0, 2.0]);

    let output = group_norm(input, Some(gamma.clone()), Some(beta.clone()), 2, 1e-5);
    let expected = group_norm(contiguous, Some(gamma), Some(beta), 2, 1e-5);

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected.into_data(),
        Tolerance::relative(1e-5).set_half_precision_relative(5e-3),
    );
}

#[test]
fn test_group_norm_rank_five_channel_innermost_dense_layout() {
    let input = TestTensor::<5>::from(TensorData::new(
        (1..=96).map(|value| value as f32).collect(),
        [2, 2, 2, 3, 4],
    ))
    .permute([0, 4, 1, 2, 3]);
    let contiguous = TestTensor::<5>::from_data(input.to_data(), &input.device());

    let output = group_norm(input, None, None, 2, 1e-5);
    let expected = group_norm(contiguous, None, None, 2, 1e-5);

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected.into_data(),
        Tolerance::relative(1e-5).set_half_precision_relative(5e-3),
    );
}

#[test]
fn test_group_norm_channel_innermost_with_spatial_permutation_fallback() {
    let input = TestTensor::<4>::from(TensorData::new(
        (1..=24).map(|value| value as f32).collect(),
        [1, 3, 2, 4],
    ))
    .permute([0, 3, 2, 1]);
    let contiguous = TestTensor::<4>::from_data(input.to_data(), &input.device());

    let output = group_norm(input, None, None, 2, 1e-5);
    let expected = group_norm(contiguous, None, None, 2, 1e-5);

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected.into_data(),
        Tolerance::relative(1e-5).set_half_precision_relative(5e-3),
    );
}

#[test]
fn test_group_norm_sliced_input() {
    let input =
        TestTensor::<3>::from([[[0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0, 7.0, 8.0]]])
            .reshape([1, 4, 3])
            .slice([0..1, 0..4, 1..3]);
    let contiguous = TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]);

    let output = group_norm(input, None, None, 2, 1e-5);
    let expected = group_norm(contiguous, None, None, 2, 1e-5);

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected.into_data(),
        Tolerance::relative(1e-5).set_half_precision_relative(5e-3),
    );
}

#[test]
fn test_group_norm_large_values() {
    let input = TestTensor::<3>::from([[[10000.0, 10008.0], [10016.0, 10024.0]]]);

    let output = group_norm(input, None, None, 1, 0.0);

    let expected = TensorData::from([[[-1.3416408, -0.4472136], [0.4472136, 1.3416408]]]);
    output.into_data().assert_approx_eq::<FloatElem>(
        &expected,
        Tolerance::relative(1e-4).set_half_precision_relative(5e-3),
    );
}

#[test]
fn test_group_norm_rank_four_and_multiple_batches() {
    let values = (1..=48).map(|value| value as f32).collect::<Vec<_>>();
    let input = TestTensor::<4>::from(TensorData::new(values.clone(), [2, 4, 2, 3]));
    let flattened = TestTensor::<3>::from(TensorData::new(values, [2, 4, 6]));

    let output = group_norm(input, None, None, 4, 1e-5).reshape([2, 4, 6]);
    let expected = group_norm(flattened, None, None, 4, 1e-5);

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected.into_data(),
        Tolerance::relative(1e-5).set_half_precision_relative(5e-3),
    );
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "rocm"))]
#[test]
fn test_group_norm_supported_float_dtypes() {
    let dtypes = if cfg!(any(feature = "cuda", feature = "rocm")) {
        alloc::vec![
            burn_tensor::DType::F16,
            burn_tensor::DType::BF16,
            burn_tensor::DType::F64,
        ]
    } else {
        alloc::vec![burn_tensor::DType::F16, burn_tensor::DType::F64]
    };

    for dtype in dtypes {
        let input =
            TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]).cast(dtype);
        let gamma = TestTensor::<1>::from([1.0, 2.0, 3.0, 4.0]).cast(dtype);
        let beta = TestTensor::<1>::from([0.0, 1.0, 2.0, 3.0]).cast(dtype);
        assert_eq!(input.dtype(), dtype);

        let output = group_norm(input, Some(gamma), Some(beta), 2, 0.0);
        assert_eq!(output.dtype(), dtype);
        let output = output.into_data().convert::<f32>();
        let expected = TensorData::from([[
            [-1.3416408, -0.4472136],
            [1.8944272, 3.6832817],
            [-2.0249224, 0.6583592],
            [4.7888546, 8.366563],
        ]]);
        output.assert_approx_eq::<f32>(&expected, Tolerance::relative(1e-2));
    }
}

#[cfg(any(feature = "cpu", feature = "cuda"))]
#[test]
fn test_group_norm_f64_preserves_precision() {
    let input =
        TestTensor::<3>::from([[[16_777_216.0, 16_777_218.0], [16_777_220.0, 16_777_222.0]]])
            .cast(burn_tensor::DType::F64);
    let input_data = TensorData::new(
        alloc::vec![16_777_216.0_f64, 16_777_218.0, 16_777_220.0, 16_777_222.0,],
        [1, 2, 2],
    );
    assert_eq!(input.dtype(), burn_tensor::DType::F64);
    input.to_data().assert_eq(&input_data, false);

    let output = group_norm(input, None, None, 1, 0.0);
    let expected = TensorData::new(
        alloc::vec![
            -1.3416407864998738_f64,
            -0.4472135954999579,
            0.4472135954999579,
            1.3416407864998738,
        ],
        [1, 2, 2],
    );

    output
        .into_data()
        .assert_approx_eq::<f64>(&expected, Tolerance::relative(1e-10));
}
