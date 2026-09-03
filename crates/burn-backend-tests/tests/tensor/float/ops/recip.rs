use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_recip_ops() {
    let data = TensorData::from([[0.5, 1.0, 2.0], [3.0, -4.0, -5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.recip();
    let expected = TensorData::from([[2.0, 1.0, 0.5], [0.33333, -0.25, -0.2]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_preserve_order_and_precision_for_shared_selected_tensor() {
    let device = Default::default();
    let values = (1..=64).map(|value| value as f32).collect::<Vec<_>>();
    let expected = values
        .chunks(16)
        .flat_map(|row| row.iter().step_by(2))
        .map(|value| value.recip())
        .collect::<Vec<_>>();
    let tensor = TestTensor::<2>::from_data(TensorData::new(values, [4, 16]), &device);
    let indices = TestTensorInt::<1>::from_data([0, 2, 4, 6, 8, 10, 12, 14], &device);
    let selected = tensor.select(1, indices);
    let shared = selected.clone();

    let output = selected.recip();
    drop(shared);

    output.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::new(expected, [4, 8]),
        Tolerance::rel_abs(1e-6, 1e-7),
    );
}
