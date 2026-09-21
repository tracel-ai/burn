use super::*;
use burn_tensor::Distribution;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn mask_fill_should_broadcast_concrete_inputs() {
    let device = Default::default();
    const N: usize = 65;
    let mask_data: Vec<bool> = (0..N * N).map(|i| i % 2 == 0).collect();

    for shape in [[1, 1], [1, N], [N, 1]] {
        for shared in [false, true] {
            let input: Vec<f32> = (0..shape[0] * shape[1]).map(|i| i as f32 + 1.0).collect();
            let expected: Vec<f32> = (0..N * N)
                .map(|i| {
                    let row = (i / N) % shape[0];
                    let col = (i % N) % shape[1];
                    if mask_data[i] {
                        -7.0
                    } else {
                        input[row * shape[1] + col]
                    }
                })
                .collect();
            let tensor = TestTensor::<2>::from_data(TensorData::new(input.clone(), shape), &device);
            let mask =
                TestTensorBool::<2>::from_data(TensorData::new(mask_data.clone(), [N, N]), &device);
            let retained_input = shared.then(|| tensor.clone());

            let output = tensor.mask_fill(mask, -7.0);

            assert_eq!(output.dims(), [N, N]);
            assert_eq!(output.into_data().try_to_vec::<f32>().unwrap(), expected);
            if let Some(tensor) = retained_input {
                assert_eq!(tensor.into_data().try_to_vec::<f32>().unwrap(), input);
            }
        }
    }
}

#[test]
fn mask_fill_should_broadcast_to_empty_output() {
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([2.0], &device);
    let mask = TestTensorBool::<1>::from_data(TensorData::new(Vec::<bool>::new(), [0]), &device);

    let output = tensor.mask_fill(mask, 7.0);

    assert_eq!(output.dims(), [0]);
    assert!(output.into_data().try_to_vec::<f32>().unwrap().is_empty());
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
