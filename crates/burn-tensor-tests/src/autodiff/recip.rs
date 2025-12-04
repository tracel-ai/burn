use crate::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_diff_recip() {
    let data = TensorData::from([2.0, 5.0, 0.4]);

    let tensor = TestAutodiffTensor::<1>::from_data(data, &Default::default()).require_grad();
    let tensor_out = tensor.clone().recip();

    let grads = tensor_out.backward();
    let grad = tensor.grad(&grads).unwrap();

    tensor_out
        .into_data()
        .assert_eq(&TensorData::from([0.5, 0.2, 2.5]), false);
    grad.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([-0.25, -0.04, -6.25]),
        Tolerance::default(),
    );
}
