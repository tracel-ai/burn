use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;
use burn_tensor::backend::Backend;

#[test]
fn test_should_mean() {
    let tensor = TestTensor::<2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.mean();
    let expected = TensorData::from([15.0 / 6.0]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_should_sum() {
    let tensor = TestTensor::<2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.sum();

    output
        .into_data()
        .assert_eq(&TensorData::from([15.0]), false);
}

#[test]
fn test_should_sum_dim_maybe_fused() {
    let tensor = TestTensor::<2>::from([[5.0], [-12.0]]);
    let tensor1 = TestTensor::<2>::from([[2.0, 3.0], [-1.0, -5.0]]);
    let ones = TestTensor::<2>::ones([2, 2], &Default::default());
    let _x = ones.clone() * tensor;
    let y = ones * tensor1;

    let output = y.clone().sum_dim(1);
    output
        .into_data()
        .assert_eq(&TensorData::from([[5.0], [-6.0]]), false);

    // Negative Indexing.
    let output = y.clone().sum_dim(-1);
    output
        .into_data()
        .assert_eq(&TensorData::from([[5.0], [-6.0]]), false);
}

#[test]
fn test_should_mean_last_dim() {
    let tensor = TestTensor::<2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.clone().mean_dim(1);
    let expected = TensorData::from([[3.0 / 3.0], [12.0 / 3.0]]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    // Negative Indexing.
    let output = tensor.clone().mean_dim(-1);
    let expected = TensorData::from([[3.0 / 3.0], [12.0 / 3.0]]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_should_sum_last_dim() {
    let tensor = TestTensor::<2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.sum_dim(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3.0], [12.0]]), false);
}

#[test]
fn test_should_sum_first_dim() {
    let tensor = TestTensor::<2>::from([[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]]);

    let output = tensor.sum_dim(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[7.0, 3.0, 5.0]]), false);
}

#[test]
fn test_should_mean_first_dim() {
    let tensor = TestTensor::<2>::from([[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]]);

    let output = tensor.mean_dim(0);

    output.into_data().assert_eq(
        &TensorData::from([[7.0 / 2.0, 3.0 / 2.0, 5.0 / 2.0]]),
        false,
    );
}

#[test]
fn test_should_sum_mid_dim_3d_non_contiguous_1() {
    let tensor = TestTensor::<3>::from([
        [[2.0, 4.0, 1.0], [7.0, -5.0, 3.0]],
        [[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]],
    ]);

    let output = tensor.swap_dims(0, 2).sum_dim(1);

    output.into_data().assert_eq(
        &TensorData::new(vec![9.0, 7.0, -1.0, 3.0, 4.0, 5.0], [3, 1, 2]),
        false,
    );
}

#[test]
fn test_should_sum_mid_dim_3d_non_contiguous_2() {
    let tensor = TestTensor::<3>::from([
        [[2.0, 4.0, 1.0], [7.0, -5.0, 3.0]],
        [[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]],
    ]);

    let output = tensor.swap_dims(0, 1).sum_dim(1);

    output.into_data().assert_eq(
        &TensorData::new(vec![5.0, 5.0, 3.0, 11.0, -3.0, 6.0], [2, 1, 3]),
        false,
    );
}

#[test]
fn test_prod_float() {
    let tensor = TestTensor::<2>::from([[2.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let output = tensor.prod();

    // 2 * 1 * 2 * 3 * 4 * 5 = 240 but we need to check the precision because of the float
    let expected = TensorData::from([240.0]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let tensor_with_zero = TestTensor::<2>::from([[2.0, 0.0, 2.0], [3.0, 4.0, 5.0]]);
    let output = tensor_with_zero.prod();

    output
        .into_data()
        .assert_eq(&TensorData::from([0.0]), false);
}

#[test]
fn test_prod_dim_float() {
    let tensor = TestTensor::<2>::from([[2.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let output = tensor.prod_dim(1);
    let expected = TensorData::from([[4.0], [60.0]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let tensor_with_zero = TestTensor::<2>::from([[2.0, 0.0, 2.0], [3.0, 4.0, 5.0]]);
    let output = tensor_with_zero.prod_dim(1);
    let expected = TensorData::from([[0.0], [60.0]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_sum_dim_2d() {
    let tensor =
        TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

    let output = tensor.clone().sum_dim(1);
    let expected = TensorData::from([[3.], [12.]]);

    output.into_data().assert_eq(&expected, false);

    let output = tensor.sum_dim(0);
    let expected = TensorData::from([[3., 5., 7.]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_sum_dims_2d() {
    let tensor =
        TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

    tensor
        .clone()
        .sum_dims(&[1])
        .to_data()
        .assert_eq(&TensorData::from([[3.], [12.]]), false);

    tensor
        .clone()
        .sum_dims(&[-1])
        .to_data()
        .assert_eq(&TensorData::from([[3.], [12.]]), false);

    tensor
        .clone()
        .sum_dims(&[0, 1])
        .to_data()
        .assert_eq(&TensorData::from([[15.]]), false);
}

#[test]
fn test_sum_and_squeeze_dims() {
    let tensor = TestTensor::<3>::from_data(
        [
            [[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]],
            [[9.0, 2.0, 5.0], [5.0, 7.0, 7.0]],
        ],
        &Default::default(),
    );

    tensor
        .sum_dims_squeeze::<1, _>(&[0, 1])
        .to_data()
        .assert_eq(&TensorData::from([20., 16., 21.]), false);
}

#[test]
fn test_sum_dim_1_reshape_maybe_fused() {
    let tensor = TestTensorInt::arange(0..9, &Default::default()).float();
    TestBackend::sync(&tensor.device()).unwrap();

    let output = tensor.reshape([3, 3]) + 2;
    let output = output.sum_dim(1);
    let expected = TensorData::from([[9.0], [18.0], [27.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_sum_dim_1_swap_dims_maybe_fused() {
    let tensor = TestTensorInt::arange(0..9, &Default::default()).float();
    let tensor = tensor.reshape([3, 3]);
    TestBackend::sync(&tensor.device()).unwrap();

    let output = tensor.swap_dims(0, 1) + 2;
    let output = output.sum_dim(1);
    let expected = TensorData::from([[15.0], [18.0], [21.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_sum_dim_2_reshape_maybe_fused_broadcast() {
    let tensor = TestTensorInt::arange(0..9, &Default::default()).float();
    TestBackend::sync(&tensor.device()).unwrap();

    let output = tensor.reshape([1, 3, 3]) + 2;
    let output = output.sum_dim(2);
    let expected = TensorData::from([[[9.0], [18.0], [27.0]]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_sum_dim_2_maybe_fused_on_write() {
    let tensor_1 = TestTensorInt::arange(0..8, &Default::default()).float();
    let tensor_2 = TestTensorInt::arange(10..12, &Default::default()).float();
    let tensor_1 = tensor_1.reshape([1, 2, 4]);
    let tensor_2 = tensor_2.reshape([1, 2, 1]);
    TestBackend::sync(&tensor_1.device()).unwrap();

    let output = (tensor_1 + tensor_2.clone()).sum_dim(2) + tensor_2;
    TestBackend::sync(&output.device()).unwrap();
    let expected = TensorData::from([[[56.0], [77.0]]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_sum_dim_3_maybe_fused_on_read_not_contiguous() {
    let tensor_1 = TestTensorInt::arange(0..8, &Default::default()).float();
    let tensor_2 = TestTensorInt::arange(16..24, &Default::default()).float();

    let tensor_1 = tensor_1.reshape([4, 2, 1]);
    let tensor_1 = tensor_1.swap_dims(0, 2);

    let tensor_2 = tensor_2.reshape([1, 4, 2]);
    let tensor_2 = tensor_2.swap_dims(1, 2);
    TestBackend::sync(&tensor_1.device()).unwrap();

    let output = (tensor_1 + tensor_2).sum_dim(2);
    let expected = TensorData::from([[[88.0], [96.0]]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_sum_dim_4_maybe_fused_on_read_not_contiguous_mixed() {
    let tensor_1 = TestTensorInt::arange(0..8, &Default::default()).float();
    let tensor_2 = TestTensorInt::arange(16..24, &Default::default()).float();
    let tensor_3 = TestTensorInt::arange(32..40, &Default::default()).float();

    let tensor_1 = tensor_1.reshape([4, 2, 1]);
    let tensor_3 = tensor_3.reshape([1, 2, 4]);
    let tensor_1 = tensor_1.swap_dims(0, 2);

    let tensor_2 = tensor_2.reshape([1, 4, 2]);
    let tensor_2 = tensor_2.swap_dims(1, 2);
    TestBackend::sync(&tensor_1.device()).unwrap();

    let output = (tensor_3 + tensor_1 + tensor_2).sum_dim(2);
    let expected = TensorData::from([[[222.0], [246.0]]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_sum_dim_5_maybe_fused_on_read_not_contiguous_mixed() {
    let tensor_1 = TestTensorInt::arange(0..8, &Default::default()).float();
    let tensor_2 = TestTensorInt::arange(16..24, &Default::default()).float();
    let tensor_3 = TestTensorInt::arange(32..40, &Default::default()).float();

    let tensor_1 = tensor_1.reshape([4, 2, 1]);
    let tensor_3 = tensor_3.reshape([1, 2, 4]);
    let tensor_1 = tensor_1.swap_dims(0, 2);

    let tensor_2 = tensor_2.reshape([1, 4, 2]);
    let tensor_2 = tensor_2.swap_dims(1, 2);
    TestBackend::sync(&tensor_1.device()).unwrap();

    let output = (tensor_3 + tensor_1 + tensor_2).sum_dim(1);
    let expected = TensorData::from([[[102.0, 112.0, 122.0, 132.0]]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_sum_dim_6_maybe_fused_on_read_not_contiguous_broadcasted() {
    let tensor_1 = TestTensorInt::arange(0..32, &Default::default()).float();
    let tensor_2 = TestTensorInt::arange(0..8, &Default::default()).float();

    let tensor_1 = tensor_1.reshape([4, 2, 2, 2]);
    let tensor_1 = tensor_1.swap_dims(3, 2);
    let tensor_1 = tensor_1.swap_dims(1, 2);

    let tensor_2 = tensor_2.reshape([1, 2, 2, 2]);

    TestBackend::sync(&tensor_1.device()).unwrap();
    let sum = tensor_2.clone().sum_dim(0);
    let sum = sum.sum_dim(1);
    let sum = sum.sum_dim(2);

    TestBackend::sync(&tensor_1.device()).unwrap();

    let _tmp = sum.clone() + 2;
    let output = (tensor_1 + tensor_2 + sum).sum_dim(1);
    let expected = TensorData::from([
        [[[29.0, 43.0], [41.0, 55.0]]],
        [[[45.0, 59.0], [57.0, 71.0]]],
        [[[61.0, 75.0], [73.0, 87.0]]],
        [[[77.0, 91.0], [89.0, 103.0]]],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_sum_dim_7_maybe_fused_on_read_reshaped() {
    let tensor_1 = TestTensorInt::arange(0..16, &Default::default()).float();

    let tensor_1 = tensor_1.reshape([4, 4]);

    TestBackend::sync(&tensor_1.device()).unwrap();

    let reshaped = tensor_1.reshape([1, 4, 4]);
    let tmp = reshaped + 5.0;
    let output = tmp.sum_dim(2);
    let expected = TensorData::from([[[26.0], [42.0], [58.0], [74.0]]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_mean_dim_fused_on_read_on_write() {
    // https://github.com/tracel-ai/burn/issues/3987
    let device = Default::default();
    let x = TestTensor::ones([128, 32, 1], &device);

    let weight = TestTensor::ones([1, 32, 1], &device);
    let options = burn_tensor::ops::ConvOptions::new([1], [0], [1], 1);
    let x = burn_tensor::module::conv1d(x, weight, None, options);
    let global = x.clone().powi_scalar(2).sum_dim(2).add_scalar(1e-5).sqrt();
    let norm = global.clone().div(global.mean_dim(1));
    let x = x.clone().mul(norm).add(x);

    let out = x.sum();

    out.into_data()
        .assert_eq(&TensorData::from([8192.0]), false);
}

#[test]
fn test_mean_dim_2d() {
    let tensor =
        TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

    let output = tensor.clone().mean_dim(1);
    let expected = TensorData::from([[1.], [4.]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let output = tensor.mean_dim(0);
    let expected = TensorData::from([[1.5, 2.5, 3.5]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_mean_dims_2d() {
    let tensor =
        TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

    tensor
        .clone()
        .mean_dims(&[1])
        .to_data()
        .assert_eq(&TensorData::from([[1.], [4.]]), false);

    tensor
        .clone()
        .mean_dims(&[-1])
        .to_data()
        .assert_eq(&TensorData::from([[1.], [4.]]), false);

    tensor
        .clone()
        .mean_dims(&[0, 1])
        .to_data()
        .assert_eq(&TensorData::from([[2.5]]), false);
}

#[test]
fn test_multiple_reduce_dims_permuted() {
    // Regression test for https://github.com/tracel-ai/burn/issues/4461.
    //
    // Also pins the f16/bf16 `mean_dim` overflow fix: after the first reduction
    // the second `mean_dim` sums 256 values that peak near 1021, so the f32
    // intermediate reaches ~261k (well above f16::MAX = 65504). A naive
    // sum-then-divide in f16 would overflow to +inf. See the sibling
    // `test_should_mean_overflow_intermediate_sum` for the scalar `.mean()`
    // code path.
    let tensor = TestTensorInt::arange(0..2 * 2 * 256, &Default::default())
        .float()
        .reshape([2, 2, 256]);

    let output = tensor
        .permute([1, 2, 0])
        .mean_dim(0)
        .mean_dim(1)
        .squeeze_dims::<1>(&[0, 1]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([255.5, 767.5]), Tolerance::default());
}

#[test]
fn test_sum_transposed() {
    // Stress the sum kernel on a non-contiguous (transposed) input. Uses
    // total reduction so the assertion doesn't depend on traversal order;
    // a stronger per-axis variant would require flex issue #4816 to be
    // resolved first.
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let output = tensor.transpose().sum();

    output
        .into_data()
        .assert_eq(&TensorData::from([21.0]), false);
}

#[test]
fn test_sum_flipped() {
    // Flip axis 0, sum along axis 1: per-row sums appear in the flipped
    // row order, so a no-op flip would give the unflipped order.
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
    let output = tensor.flip([0]).sum_dim(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[60.0], [6.0]]), false);
}

#[test]
fn test_sum_dim_flipped() {
    // Flip axis 0, reduce axis 1: rows are swapped so per-row sums appear
    // reversed, which a flip-as-noop would not produce.
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let output = tensor.flip([0]).sum_dim(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[15.0], [6.0]]), false);
}

#[test]
fn test_sum_dim_flipped_axis1() {
    // Flip axis 1, reduce axis 0: columns are reversed, so column sums
    // appear in reversed order.
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let output = tensor.flip([1]).sum_dim(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[9.0, 7.0, 5.0]]), false);
}

#[test]
fn test_mean_dim_flipped() {
    // Flip axis 0, mean axis 1: row means appear in reversed row order.
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let output = tensor.flip([0]).mean_dim(1);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([[5.0], [2.0]]), Tolerance::default());
}

#[test]
fn test_prod_flipped() {
    // Flip axis 0, prod along axis 1: per-row products appear in reversed
    // row order.
    let tensor = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);
    let output = tensor.flip([0]).prod_dim(1);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([[12.0], [2.0]]), Tolerance::default());
}

#[test]
fn test_sum_narrowed() {
    // Narrow to indices 1..4 of [0, 1, 2, 3, 4] -> [1, 2, 3]. Without the
    // narrow the sum would be 10, not 6.
    let tensor = TestTensor::<1>::from([0.0, 1.0, 2.0, 3.0, 4.0]);
    let output = tensor.narrow(0, 1, 3).sum();

    output
        .into_data()
        .assert_eq(&TensorData::from([6.0]), false);
}

#[test]
fn test_sum_flipped_both_axes() {
    // Flip both axes, sum along axis 0: column sums appear in reversed
    // order because axis 1 was flipped; row pairing is also swapped so a
    // missing axis-0 flip would give different pairings.
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let output = tensor.flip([0, 1]).sum_dim(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[9.0, 7.0, 5.0]]), false);
}

#[test]
fn test_sum_dim_4d_middle_dim() {
    // Regression: 4D tensor reducing a middle dim (shape [1, 84, 80, 80]).
    // Fill with 1.0 so every output position should sum to 84.0.
    let shape = [1, 84, 80, 80];
    let n: usize = shape.iter().product();
    let data: Vec<f32> = vec![1.0; n];
    let tensor = TestTensor::<4>::from_data(TensorData::new(data, shape), &Default::default());

    let output = tensor.sum_dim(1);

    let expected = TensorData::new(vec![84.0f32; 1 * 1 * 80 * 80], [1, 1, 80, 80]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_should_mean_overflow_intermediate_sum() {
    let tensor = TestTensorInt::arange(0..1024, &Default::default()).float();
    let output = tensor.mean();
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([511.5]), Tolerance::default());
}
