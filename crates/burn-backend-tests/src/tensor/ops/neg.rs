use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_neg_ops() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.neg();
    let expected = TensorData::from([[-0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]]).convert::<f32>();

    // -0.0 is represented differently than 0.0 so we make sure the values are the same in f32
    assert_eq!(
        output
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap(),
        expected.as_slice::<f32>().unwrap()
    );
}
