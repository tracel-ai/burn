use super::*;
use burn_tensor::Distribution;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn mask_fill_should_broadcast_to_empty_output() {
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([2.0], &device);
    let mask = TestTensorBool::<1>::from_data(TensorData::new(Vec::<bool>::new(), [0]), &device);

    let output = tensor.mask_fill(mask, 7.0);

    assert_eq!(output.dims(), [0]);
    output
        .into_data()
        .assert_eq(&TensorData::new(Vec::<f32>::new(), [0]), false);
}

#[test]
fn mask_fill_should_match_reference_backend() {
    let (tensor, mask, tensor_ref, mask_ref) = inputs_mask_fill();

    // MaskFillStrategy::Readonly
    let _clone_for_readonly = tensor.clone();

    let actual = tensor.mask_fill(mask, 4.0);
    let expected = tensor_ref.mask_fill(mask_ref, 4.0);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

#[test]
fn mask_fill_inplace_should_match_reference_backend() {
    let (tensor, mask, tensor_ref, mask_ref) = inputs_mask_fill();

    // MaskFillStrategy::Inplace
    let actual = tensor.mask_fill(mask, 4.0);
    let expected = tensor_ref.mask_fill(mask_ref, 4.0);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

#[allow(clippy::type_complexity)]
fn inputs_mask_fill() -> (
    TestTensor<3>,
    TestTensorBool<3>,
    TestTensor<3>,
    TestTensorBool<3>,
) {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<3>::random([2, 6, 256], Distribution::Default, &device);
    let mask = TestTensor::<3>::random([2, 6, 256], Distribution::Uniform(0., 1.), &device)
        .lower_equal_elem(0.5);

    let tensor_ref = TestTensor::<3>::from_data(tensor.to_data(), &ref_device);
    let mask_ref = TestTensorBool::<3>::from_data(mask.to_data(), &ref_device);

    (tensor, mask, tensor_ref, mask_ref)
}

const BROADCAST_DIM: usize = 65;

#[test]
fn mask_fill_should_broadcast_scalar_to_larger_mask() {
    let (tensor, mask, expected) = inputs_mask_fill_broadcast([1, 1]);

    let output = tensor.mask_fill(mask, -7.0);

    assert_eq!(output.dims(), [BROADCAST_DIM, BROADCAST_DIM]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn mask_fill_should_broadcast_shared_scalar_to_larger_mask() {
    let (tensor, mask, expected) = inputs_mask_fill_broadcast([1, 1]);
    let original = tensor.to_data();

    let output = tensor.clone().mask_fill(mask, -7.0);

    assert_eq!(output.dims(), [BROADCAST_DIM, BROADCAST_DIM]);
    output.into_data().assert_eq(&expected, false);
    tensor.into_data().assert_eq(&original, false);
}

#[test]
fn mask_fill_should_broadcast_row_to_larger_mask() {
    let (tensor, mask, expected) = inputs_mask_fill_broadcast([1, BROADCAST_DIM]);

    let output = tensor.mask_fill(mask, -7.0);

    assert_eq!(output.dims(), [BROADCAST_DIM, BROADCAST_DIM]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn mask_fill_should_broadcast_shared_row_to_larger_mask() {
    let (tensor, mask, expected) = inputs_mask_fill_broadcast([1, BROADCAST_DIM]);
    let original = tensor.to_data();

    let output = tensor.clone().mask_fill(mask, -7.0);

    assert_eq!(output.dims(), [BROADCAST_DIM, BROADCAST_DIM]);
    output.into_data().assert_eq(&expected, false);
    tensor.into_data().assert_eq(&original, false);
}

#[test]
fn mask_fill_should_broadcast_column_to_larger_mask() {
    let (tensor, mask, expected) = inputs_mask_fill_broadcast([BROADCAST_DIM, 1]);

    let output = tensor.mask_fill(mask, -7.0);

    assert_eq!(output.dims(), [BROADCAST_DIM, BROADCAST_DIM]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn mask_fill_should_broadcast_shared_column_to_larger_mask() {
    let (tensor, mask, expected) = inputs_mask_fill_broadcast([BROADCAST_DIM, 1]);
    let original = tensor.to_data();

    let output = tensor.clone().mask_fill(mask, -7.0);

    assert_eq!(output.dims(), [BROADCAST_DIM, BROADCAST_DIM]);
    output.into_data().assert_eq(&expected, false);
    tensor.into_data().assert_eq(&original, false);
}

fn inputs_mask_fill_broadcast(shape: [usize; 2]) -> (TestTensor<2>, TestTensorBool<2>, TensorData) {
    let device = Default::default();
    let input: Vec<f32> = (0..shape[0] * shape[1]).map(|i| i as f32 + 1.0).collect();
    let mask_data: Vec<bool> = (0..BROADCAST_DIM * BROADCAST_DIM)
        .map(|i| i % 2 == 0)
        .collect();
    let expected: Vec<f32> = mask_data
        .iter()
        .enumerate()
        .map(|(i, &masked)| {
            let row = (i / BROADCAST_DIM) % shape[0];
            let col = (i % BROADCAST_DIM) % shape[1];
            if masked {
                -7.0
            } else {
                input[row * shape[1] + col]
            }
        })
        .collect();
    let tensor = TestTensor::<2>::from_data(TensorData::new(input, shape), &device);
    let mask = TestTensorBool::<2>::from_data(
        TensorData::new(mask_data, [BROADCAST_DIM, BROADCAST_DIM]),
        &device,
    );
    let expected = TensorData::new(expected, [BROADCAST_DIM, BROADCAST_DIM]);
    (tensor, mask, expected)
}
