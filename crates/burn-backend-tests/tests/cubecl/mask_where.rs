use super::*;
use burn_tensor::Device;
use burn_tensor::Distribution;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn mask_where_should_broadcast_rows_and_columns() {
    let device = Device::default();
    const N: usize = 65;
    let row: Vec<f32> = (0..N).map(|i| i as f32 + 1.0).collect();
    let column: Vec<f32> = (0..N).map(|i| i as f32 + 101.0).collect();
    let mask: Vec<bool> = (0..N).map(|i| i % 2 == 0).collect();
    let expected: Vec<f32> = column
        .iter()
        .flat_map(|&value| {
            row.iter()
                .zip(&mask)
                .map(move |(&input, &m)| if m { value } else { input })
        })
        .collect();
    let expected = TensorData::new(expected, [N, N]);
    let tensor = TestTensor::<2>::from_data(TensorData::new(row, [1, N]), &device);
    let value = TestTensor::<2>::from_data(TensorData::new(column, [N, 1]), &device);
    let mask = TestTensorBool::<2>::from_data(TensorData::new(mask, [1, N]), &device);

    let output = tensor.mask_where(mask, value);

    assert_eq!(output.dims(), [N, N]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn mask_where_should_match_reference_backend() {
    let (tensor, value, mask, tensor_ref, value_ref, mask_ref) = inputs_mask_where();

    let actual = tensor.mask_where(mask, value);
    let expected = tensor_ref.mask_where(mask_ref, value_ref);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}
#[test]
fn mask_where_inplace_lhs_should_match_reference_backend() {
    let (tensor, value, mask, tensor_ref, value_ref, mask_ref) = inputs_mask_where();

    // MaskWhereStrategy::InplaceLhs
    let actual = tensor.mask_where(mask, value);
    let expected = tensor_ref.mask_where(mask_ref, value_ref);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

#[test]
fn mask_where_inplace_rhs_should_match_reference_backend() {
    let (tensor, value, mask, tensor_ref, value_ref, mask_ref) = inputs_mask_where();

    // MaskWhereStrategy::InplaceRhs
    let _clone_for_inplace_rhs = tensor.clone();

    let actual = tensor.mask_where(mask, value);
    let expected = tensor_ref.mask_where(mask_ref, value_ref);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

#[allow(clippy::type_complexity)]
fn inputs_mask_where() -> (
    TestTensor<3>,
    TestTensor<3>,
    TestTensorBool<3>,
    TestTensor<3>,
    TestTensor<3>,
    TestTensorBool<3>,
) {
    let device = Device::default();
    let ref_device = ReferenceDevice::new();

    device.seed(0);

    let tensor = TestTensor::<3>::random([2, 6, 256], Distribution::Default, &device);
    let value = TestTensor::<3>::random([2, 6, 256], Distribution::Default, &device);
    let mask = TestTensor::<3>::random([2, 6, 256], Distribution::Uniform(0., 1.), &device)
        .lower_equal_elem(0.5);

    let tensor_ref = TestTensor::<3>::from_data(tensor.to_data(), &ref_device);
    let value_ref = TestTensor::<3>::from_data(value.to_data(), &ref_device);
    let mask_ref = TestTensorBool::<3>::from_data(mask.to_data(), &ref_device);
    mask.to_data().assert_eq(&mask_ref.to_data(), false);

    (tensor, value, mask, tensor_ref, value_ref, mask_ref)
}

// Exceed a single workgroup so sizing the launch from the scalar input cannot suffice.
const BROADCAST_LEN: usize = 4097;

#[test]
fn mask_where_should_broadcast_scalar_inputs() {
    let (tensor, mask, value, expected) = inputs_mask_where_broadcast(1);

    let output = tensor.mask_where(mask, value);

    assert_eq!(output.dims(), [BROADCAST_LEN]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn mask_where_should_broadcast_scalar_inputs_with_shared_input() {
    let (tensor, mask, value, expected) = inputs_mask_where_broadcast(1);

    let output = tensor.clone().mask_where(mask, value);

    assert_eq!(output.dims(), [BROADCAST_LEN]);
    output.into_data().assert_eq(&expected, false);
    tensor
        .into_data()
        .assert_eq(&TensorData::from([2.0]), false);
}

#[test]
fn mask_where_should_broadcast_shared_scalar_inputs() {
    let (tensor, mask, value, expected) = inputs_mask_where_broadcast(1);

    let output = tensor.clone().mask_where(mask, value.clone());

    assert_eq!(output.dims(), [BROADCAST_LEN]);
    output.into_data().assert_eq(&expected, false);
    tensor
        .into_data()
        .assert_eq(&TensorData::from([2.0]), false);
    value.into_data().assert_eq(&TensorData::from([7.0]), false);
}

#[test]
fn mask_where_should_broadcast_scalar_input_with_full_sized_value() {
    let (tensor, mask, value, expected) = inputs_mask_where_broadcast(BROADCAST_LEN);

    let output = tensor.mask_where(mask, value);

    assert_eq!(output.dims(), [BROADCAST_LEN]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn mask_where_should_broadcast_shared_scalar_input_with_full_sized_value() {
    let (tensor, mask, value, expected) = inputs_mask_where_broadcast(BROADCAST_LEN);

    let output = tensor.clone().mask_where(mask, value);

    assert_eq!(output.dims(), [BROADCAST_LEN]);
    output.into_data().assert_eq(&expected, false);
    tensor
        .into_data()
        .assert_eq(&TensorData::from([2.0]), false);
}

#[test]
fn mask_where_should_broadcast_shared_scalar_input_with_shared_full_sized_value() {
    let (tensor, mask, value, expected) = inputs_mask_where_broadcast(BROADCAST_LEN);

    let output = tensor.clone().mask_where(mask, value.clone());

    assert_eq!(output.dims(), [BROADCAST_LEN]);
    output.into_data().assert_eq(&expected, false);
    tensor
        .into_data()
        .assert_eq(&TensorData::from([2.0]), false);
    value.into_data().assert_eq(
        &TensorData::new(vec![7.0; BROADCAST_LEN], [BROADCAST_LEN]),
        false,
    );
}

fn inputs_mask_where_broadcast(
    value_len: usize,
) -> (TestTensor<1>, TestTensorBool<1>, TestTensor<1>, TensorData) {
    let device = Device::default();
    let mask_data: Vec<bool> = (0..BROADCAST_LEN).map(|i| i % 2 == 0).collect();
    let expected: Vec<f32> = mask_data
        .iter()
        .map(|&m| if m { 7.0 } else { 2.0 })
        .collect();
    let tensor = TestTensor::<1>::from_data([2.0], &device);
    let value =
        TestTensor::<1>::from_data(TensorData::new(vec![7.0; value_len], [value_len]), &device);
    let mask = TestTensorBool::<1>::from_data(TensorData::new(mask_data, [BROADCAST_LEN]), &device);
    let expected = TensorData::new(expected, [BROADCAST_LEN]);
    (tensor, mask, value, expected)
}
