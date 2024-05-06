#[burn_tensor_testgen::testgen(ad_select)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Int, Tensor};

    #[test]
    fn test_select_grad() {
        let device = Default::default();
        let tensor_1 =
            TestAutodiffTensor::from_data(Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]), &device)
                .require_grad();
        let indices = Tensor::<TestAutodiffBackend, 1, Int>::from_data(Data::from([1, 0]), &device);

        let tensor_2 = tensor_1.clone().matmul(tensor_1.clone().transpose());
        let tensor_3 = tensor_1.clone().select(0, indices);
        let tensor_4 = tensor_2.matmul(tensor_3);

        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();

        assert_eq!(
            grad_1.into_data(),
            Data::from([[109., 148., 187.], [37., 58., 79.]])
        );
    }

    #[test]
    fn test_select_assign_grad() {
        let device = Default::default();
        let tensor_1 =
            TestAutodiffTensor::from_data(Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]), &device)
                .require_grad();
        let values =
            TestAutodiffTensor::from_data(Data::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), &device)
                .require_grad();
        let indices = Tensor::<TestAutodiffBackend, 1, Int>::from_data(Data::from([1, 0]), &device);

        let tensor_2 = tensor_1.clone().matmul(tensor_1.clone().transpose());
        let tensor_3 = tensor_1.clone().select_assign(0, indices, values.clone());
        let tensor_4 = tensor_2.matmul(tensor_3);

        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = values.grad(&grads).unwrap();

        assert_eq!(
            grad_1.into_data(),
            Data::from([[127., 199., 271.], [172., 244., 316.]])
        );
        assert_eq!(
            grad_2.into_data(),
            Data::from([[64., 64., 64.], [19., 19., 19.]])
        );
    }

    #[test]
    fn test_select_assign_grad_different_shapes() {
        let device = Default::default();

        let indices: Tensor<TestAutodiffBackend, 1, Int> = Tensor::from_ints([1], &device);
        let x: Tensor<TestAutodiffBackend, 2> = Tensor::ones([1, 1], &device).require_grad();
        let y = Tensor::ones([2, 1], &device).require_grad();

        let w = y.clone().select_assign(0, indices, x.clone());
        let w = w.matmul(y.clone().transpose());

        let grads = w.backward();
        let x_grad = x.grad(&grads).unwrap();
        let y_grad = y.grad(&grads).unwrap();

        assert_eq!(x_grad.into_data(), Data::from([[2.0]]));
        assert_eq!(y_grad.into_data(), Data::from([[5.0], [5.0]]));
    }
}
