use super::*;
use burn_tensor::Device;
use burn_tensor::Distribution;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn mask_where_should_broadcast_concrete_inputs() {
    let device = Device::default();
    // Exceed a single workgroup so sizing the launch from the scalar input cannot suffice.
    const N: usize = 4097;
    let mask_data: Vec<bool> = (0..N).map(|i| i % 2 == 0).collect();
    let expected: Vec<f32> = mask_data
        .iter()
        .map(|&m| if m { 7.0 } else { 2.0 })
        .collect();

    for value_len in [1, N] {
        // Exercise both in-place candidates and the read-only path. A full-sized value may
        // still be reused, but the work must cover its entire length.
        for (share_input, share_value) in [(false, false), (true, false), (true, true)] {
            let tensor = TestTensor::<1>::from_data([2.0], &device);
            let value = TestTensor::<1>::from_data(
                TensorData::new(vec![7.0; value_len], [value_len]),
                &device,
            );
            let mask =
                TestTensorBool::<1>::from_data(TensorData::new(mask_data.clone(), [N]), &device);
            let retained_input = share_input.then(|| tensor.clone());
            let retained_value = share_value.then(|| value.clone());

            let output = tensor.mask_where(mask, value);

            assert_eq!(output.dims(), [N]);
            assert_eq!(output.into_data().try_to_vec::<f32>().unwrap(), expected);
            if let Some(tensor) = retained_input {
                tensor
                    .into_data()
                    .assert_eq(&TensorData::from([2.0]), false);
            }
            if let Some(value) = retained_value {
                assert_eq!(
                    value.into_data().try_to_vec::<f32>().unwrap(),
                    vec![7.0; value_len]
                );
            }
        }
    }
}

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
    let tensor = TestTensor::<2>::from_data(TensorData::new(row, [1, N]), &device);
    let value = TestTensor::<2>::from_data(TensorData::new(column, [N, 1]), &device);
    let mask = TestTensorBool::<2>::from_data(TensorData::new(mask, [1, N]), &device);

    let output = tensor.mask_where(mask, value);

    assert_eq!(output.dims(), [N, N]);
    assert_eq!(output.into_data().try_to_vec::<f32>().unwrap(), expected);
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
