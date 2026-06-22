use super::*;
use burn_tensor::{Element, Shape, TensorData};

#[test]
fn should_support_float_dtype() {
    let tensor = TestTensor::<2>::from([[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]])/*.into_primitive()*/;

    assert_eq!(tensor.shape(), Shape::new([2, 3]));
    assert_eq!(
        tensor.dtype(),
        FloatElem::dtype() // default float elem type
    );
}

#[test]
fn should_support_into_data_from_data() {
    let device = Default::default();
    let data =
        TestTensor::<2>::from_data([[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]], &device).into_data();
    let tensor = TestTensor::<2>::from_data(data, &device).slice(0);

    // Regression test for `LazyDeviceController` from_data(tensor.into_data()) roundtrips
    // These unnecessary round-trips should be avoided, but should not panic
    tensor
        .into_data()
        .assert_eq(&TensorData::from([[0.0, -1.0, 2.0]]), false);
}
