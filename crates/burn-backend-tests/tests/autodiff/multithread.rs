use super::*;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn should_behave_the_same_with_multithread() {
    let data_1 = TensorData::from([[1.0, 7.0], [13.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);

    let with_move = || {
        let device = Default::default();
        let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1.clone(), &device).require_grad();
        let tensor_2 = TestAutodiffTensor::from_data(data_2.clone(), &device).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = tensor_3.clone().matmul(tensor_2.clone());
        let tensor_5 = tensor_4.matmul(tensor_3);

        // Task 1
        let tensor_1_cloned = tensor_1.clone();
        let tensor_2_cloned = tensor_2.clone();
        let tensor_5_cloned = tensor_5.clone();

        let first_call = move || {
            let tensor_6_1 = tensor_5_cloned.matmul(tensor_2_cloned);
            tensor_6_1.matmul(tensor_1_cloned)
        };

        // Task 2
        let tensor_1_cloned = tensor_1.clone();
        let tensor_2_cloned = tensor_2.clone();
        let tensor_5_cloned = tensor_5;

        let second_call = move || {
            let tensor_6_2 = tensor_5_cloned.matmul(tensor_1_cloned);
            tensor_6_2.matmul(tensor_2_cloned)
        };

        let tensor_7_1_handle = std::thread::spawn(first_call);
        let tensor_7_2_handle = std::thread::spawn(second_call);

        let tensor_7_1 = tensor_7_1_handle.join().unwrap();
        let tensor_7_2 = tensor_7_2_handle.join().unwrap();
        let tensor_8 = tensor_7_1.matmul(tensor_7_2);

        let grads = tensor_8.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        (grad_1, grad_2)
    };
    let without_move = || {
        let device = Default::default();
        let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1.clone(), &device).require_grad();
        let tensor_2 = TestAutodiffTensor::from_data(data_2.clone(), &device).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = tensor_3.clone().matmul(tensor_2.clone());
        let tensor_5 = tensor_4.matmul(tensor_3);

        // Task 1
        let tensor_6_1 = tensor_5.clone().matmul(tensor_2.clone());
        let tensor_7_1 = tensor_6_1.matmul(tensor_1.clone());

        // Task 2
        let tensor_6_2 = tensor_5.matmul(tensor_1.clone());
        let tensor_7_2 = tensor_6_2.matmul(tensor_2.clone());

        let tensor_8 = tensor_7_1.matmul(tensor_7_2);

        let grads = tensor_8.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        (grad_1, grad_2)
    };

    let (grad_1, grad_2) = without_move();
    let (grad_1_moved, grad_2_moved) = with_move();

    grad_1
        .into_data()
        .assert_approx_eq::<FloatElem>(&grad_1_moved.into_data(), Tolerance::default());
    grad_2
        .into_data()
        .assert_approx_eq::<FloatElem>(&grad_2_moved.into_data(), Tolerance::default());
}
