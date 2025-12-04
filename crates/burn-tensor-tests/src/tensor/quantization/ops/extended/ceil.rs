use crate::qtensor::*;
use crate::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_ceil_ops() {
    let tensor =
        QTensor::<TestBackend, 2>::int8([[24.0423, 87.9478, 76.1838], [59.6929, 43.8169, 94.8826]]);

    let output = tensor.ceil();
    let expected = TensorData::from([[25., 88., 77.], [60., 44., 96.]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(1e-1, 1e-1));
}
