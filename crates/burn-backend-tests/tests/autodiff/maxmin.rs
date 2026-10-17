use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_diff_max_dim() {
    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data([[1.0, 7.0], [-2.0, -3.0]], &device).require_grad();
    let tensor_2 = TestTensor::from_data([[4.0, -7.0], [2.0, 3.0]], &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = tensor_1.clone().mul(tensor_3.max_dim(1).unsqueeze());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([[50.0, 34.0], [40.0, -10.0]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([[8.0, 10.0], [56.0, 15.0]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_min_dim() {
    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data([[1.0, 7.0], [-2.0, -3.0]], &device).require_grad();
    let tensor_2 = TestTensor::from_data([[4.0, -7.0], [2.0, 3.0]], &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = tensor_1.clone().mul(tensor_3.min_dim(1).unsqueeze());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([[-42.0, 38.0], [-34.0, -24.0]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([[10.0, 8.0], [15.0, 56.0]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_min_dim_3d_dim1() {
    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<3>::from_data([[[1.0, 7.0], [-2.0, -3.0]]], &device).require_grad();
    let tensor_2 = TestTensor::<3>::from_data([[[4., -7.], [2., 3.]]], &device).require_grad();

    let tensor_3 = tensor_1.clone().mul(tensor_2.clone());
    let tensor_4 = tensor_3.min_dim(1);

    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([[[0., -7.], [2., 0.]]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([[[0., 7.], [-2., -0.]]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

/// `max_dim_with_indices` and `min_dim_with_indices` register their own backward and must
/// route gradients like `max_dim` and `min_dim`, ties included.
#[test]
fn dim_with_indices_grads_match_dim() {
    let device = AutodiffDevice::new();
    let data = TensorData::from([
        [1.0, 5.0, 5.0, 2.0],
        [7.0, -1.0, 7.0, 0.0],
        [3.0, 3.0, 3.0, 3.0],
    ]);

    for dim in [0, 1] {
        for is_max in [true, false] {
            let grad = |with_indices: bool| {
                let x = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
                let y = match (is_max, with_indices) {
                    (true, false) => x.clone().max_dim(dim),
                    (true, true) => x.clone().max_dim_with_indices(dim).0,
                    (false, false) => x.clone().min_dim(dim),
                    (false, true) => x.clone().min_dim_with_indices(dim).0,
                };
                let weights =
                    TestTensorInt::<1>::arange(1..y.shape().num_elements() as i64 + 1, &device)
                        .float()
                        .reshape::<2, _>(y.shape());
                let grads = (y * weights).sum().backward();
                x.grad(&grads).unwrap().into_data()
            };
            grad(true).assert_eq(&grad(false), false);
        }
    }
}
