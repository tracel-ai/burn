use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_diff_cos() {
    let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[6.0, 7.0], [9.0, 10.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().cos());
    let tensor_4 = tensor_3.matmul(tensor_2.clone());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    // Metal has less precise trigonometric functions
    let tolerance = Tolerance::default().set_half_precision_relative(1e-2);

    grad_1.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[26.8063, -27.7870], [26.8063, -27.7870]]),
        tolerance,
    );

    grad_2.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[9.222064, -39.123375], [-28.721354, 49.748356]]),
        tolerance,
    );
}

#[test]
fn should_diff_sin() {
    let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[6.0, 7.0], [9.0, 10.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().sin());
    let tensor_4 = tensor_3.matmul(tensor_2.clone());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    // Metal has less precise trigonometric functions
    let tolerance = Tolerance::default().set_half_precision_relative(1e-2);

    let expected = TensorData::from([[8.8500, -4.9790], [8.8500, -4.9790]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    let expected = TensorData::from([[38.668987, 44.194775], [-59.97261, -80.46094]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn should_diff_tanh() {
    let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[6.0, 7.0], [9.0, 10.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().tanh());
    let tensor_4 = tensor_3.matmul(tensor_2.clone());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let tolerance = Tolerance::default().set_half_precision_relative(8e-3);
    let expected = TensorData::from([[32.0, 32.0], [32.0, 32.0]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    let expected = TensorData::from([[8.00092, 8.000153], [8.000003, 7.999995]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn should_diff_cosh() {
    let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[0.5, 1.0], [1.5, 2.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().cosh());
    let tensor_4 = tensor_3.matmul(tensor_2.clone());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[7.092221, 16.696301], [7.092221, 16.696301]]),
        Tolerance::default(),
    );

    grad_2.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[17.489855, 27.484539], [39.409813, 86.910278]]),
        Tolerance::default(),
    );
}

#[test]
fn should_diff_sinh() {
    let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[0.5, 1.0], [1.5, 2.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().sinh());
    let tensor_4 = tensor_3.matmul(tensor_2.clone());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[4.894847, 15.887931], [4.894847, 15.887931]]),
        Tolerance::default(),
    );

    grad_2.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[17.284000, 28.412029], [39.302979, 87.498329]]),
        Tolerance::default(),
    );
}

#[test]
fn should_diff_tan() {
    let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[0.5, 1.0], [0.3, 0.8]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().tan());
    let tensor_4 = tensor_3.matmul(tensor_2.clone());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[2.532602, 1.596607], [2.532602, 1.596607]]),
        Tolerance::default(),
    );

    grad_2.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[9.028598, 14.489801], [18.038082, 21.151270]]),
        Tolerance::default(),
    );
}

#[test]
fn should_diff_asin() {
    let data_1 = TensorData::from([[0.0, 0.1], [0.3, 0.4]]);
    let data_2 = TensorData::from([[0.2, 0.3], [0.5, 0.6]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().asin());
    let tensor_4 = tensor_3.matmul(tensor_2.clone());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[0.435841, 0.969651], [0.435841, 0.969651]]),
        Tolerance::default(),
    );

    grad_2.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[0.475300, 0.668141], [0.701834, 1.100658]]),
        Tolerance::default(),
    );
}

#[test]
fn should_diff_acos() {
    let data_1 = TensorData::from([[0.0, 0.1], [0.3, 0.4]]);
    let data_2 = TensorData::from([[0.2, 0.3], [0.5, 0.6]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().acos());
    let tensor_4 = tensor_3.matmul(tensor_2.clone());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[2.077433, 1.543624], [2.077433, 1.543624]]),
        Tolerance::default(),
    );

    grad_2.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[0.781337, 0.588496], [0.554804, 0.155979]]),
        Tolerance::default(),
    );
}

#[test]
fn should_diff_atan() {
    let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[0.5, 1.0], [1.5, 2.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().atan());
    let tensor_4 = tensor_3.matmul(tensor_2.clone());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[3.444365, 5.349211], [3.444365, 5.349211]]),
        Tolerance::default(),
    );

    grad_2.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[9.904911, 11.554912], [10.199631, 11.391938]]),
        Tolerance::default(),
    );
}

#[test]
fn should_diff_asinh() {
    let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[0.5, 1.0], [1.5, 2.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().asinh());
    let tensor_4 = tensor_3.matmul(tensor_2.clone());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[3.806625, 6.844869], [3.806625, 6.844869]]),
        Tolerance::default(),
    );

    grad_2.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[11.442373, 14.842072], [14.022551, 17.688538]]),
        Tolerance::default(),
    );
}

#[test]
fn should_diff_acosh() {
    let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[1.5, 2.0], [2.5, 3.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().acosh());
    let tensor_4 = tensor_3.matmul(tensor_2.clone());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[10.611752, 15.178907], [10.611752, 15.178907]]),
        Tolerance::default(),
    );

    grad_2.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[20.112753, 20.247547], [20.402235, 22.487328]]),
        Tolerance::default(),
    );
}

#[test]
fn should_diff_atanh() {
    let data_1 = TensorData::from([[0.0, 0.1], [0.3, 0.4]]);
    let data_2 = TensorData::from([[0.2, 0.3], [0.5, 0.6]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().atanh());
    let tensor_4 = tensor_3.matmul(tensor_2.clone());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[0.441838, 1.037115], [0.441838, 1.037115]]),
        Tolerance::default(),
    );

    grad_2.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[0.491723, 0.698110], [0.772763, 1.298805]]),
        Tolerance::default(),
    );
}

#[test]
fn should_diff_atan2() {
    let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[0.5, 1.0], [1.5, 2.0]]);
    let data_3 = TensorData::from([[1.0, 0.5], [2.0, 1.5]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();
    let tensor_3 = TestAutodiffTensor::from_data(data_3, &device).require_grad();

    let tensor_4 = tensor_1
        .clone()
        .matmul(tensor_2.clone().atan2(tensor_3.clone()));
    let tensor_5 = tensor_4.matmul(tensor_2.clone());
    let grads = tensor_5.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();
    let grad_3 = tensor_3.grad(&grads).unwrap();

    grad_1.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[4.570492, 4.210785], [4.570492, 4.210785]]),
        Tolerance::default(),
    );

    grad_2.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[8.208448, 8.808449], [10.357923, 12.157923]]),
        Tolerance::default(),
    );

    grad_3.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[-1.8, -8.4], [-1.8, -5.6]]),
        Tolerance::default(),
    );
}
