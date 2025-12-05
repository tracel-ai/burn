use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_softplus_d2() {
    let tensor = TestTensor::<2>::from([[-0.4240, -0.9574, -0.2215], [-0.5767, 0.7218, -0.1620]]);

    let output = activation::softplus(tensor.clone(), 1.0);
    let expected = TensorData::from([
        [0.503453, 0.324898, 0.588517],
        [0.445806, 1.117805, 0.615424],
    ]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let output = activation::softplus(tensor, 2.0);
    let expected = TensorData::from([
        [0.178232, 0.068737, 0.247990],
        [0.137132, 0.827771, 0.272106],
    ]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
