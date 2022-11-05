use super::super::TestADTensor;
use burn_tensor::Data;

#[test]
fn test_div_grad() {
    let data_1 = Data::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = Data::from([[6.0, 7.0], [9.0, 10.0]]);

    let tensor_1 = TestADTensor::from_data(data_1);
    let tensor_2 = TestADTensor::from_data(data_2);

    let tensor_3 = tensor_1.matmul(&tensor_2);
    let tensor_4 = tensor_3.div(&tensor_2);

    let grads = tensor_4.sum().backward();
    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_approx_eq(&Data::from([[2.00, 2.9286], [1.3667, 2.0]]), 3);
    grad_2
        .to_data()
        .assert_approx_eq(&Data::from([[0.0833, 0.0959], [-0.0556, -0.0671]]), 3);
}
