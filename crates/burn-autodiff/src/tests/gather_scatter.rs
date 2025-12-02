#[burn_tensor_testgen::testgen(ad_gather_scatter)]
mod tests {
    use super::*;
    use burn_tensor::{IndexingUpdateOp, Int, Tensor, TensorData};

    #[test]
    fn test_gather_grad() {
        let device = Default::default();
        let tensor_1 = TestAutodiffTensor::from_data(
            TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            &device,
        )
        .require_grad();
        let indices = Tensor::<TestAutodiffBackend, 2, Int>::from_data(
            TensorData::from([[2, 1, 0, 1, 2], [1, 0, 2, 1, 0]]),
            &device,
        );

        let tensor_2 = tensor_1.clone().matmul(tensor_1.clone().transpose());
        let tensor_3 = tensor_1.clone().gather(1, indices);
        let tensor_4 = tensor_2.matmul(tensor_3);

        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();

        grad_1.to_data().assert_eq(
            &TensorData::from([[94., 150., 187.], [242., 305., 304.]]),
            false,
        );
    }

    #[test]
    fn test_scatter_grad() {
        let device = Default::default();
        let tensor_1 = TestAutodiffTensor::from_data(
            TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            &device,
        )
        .require_grad();
        let values = TestAutodiffTensor::from_data(
            TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        )
        .require_grad();
        let indices = Tensor::<TestAutodiffBackend, 2, Int>::from_data(
            TensorData::from([[2, 1, 0], [2, 0, 1]]),
            &device,
        );

        let tensor_2 = tensor_1.clone().matmul(tensor_1.clone().transpose());
        let tensor_3 = tensor_1
            .clone()
            .scatter(1, indices, values.clone(), IndexingUpdateOp::Add);
        let tensor_4 = tensor_2.matmul(tensor_3);

        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = values.grad(&grads).unwrap();

        grad_1.to_data().assert_eq(
            &TensorData::from([[127., 181., 235.], [226., 316., 406.]]),
            false,
        );
        grad_2
            .to_data()
            .assert_eq(&TensorData::from([[19., 19., 19.], [64., 64., 64.]]), false);
    }

    #[test]
    fn test_scatter_add_grad_partial_indices() {
        let device = Default::default();
        let tensor_1 = TestAutodiffTensor::from_data(
            TensorData::from([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]),
            &device,
        )
        .require_grad();
        let tensor_2 = TestAutodiffTensor::from_data(
            TensorData::from([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
            &device,
        )
        .require_grad();
        let values = TestAutodiffTensor::from_data(TensorData::from([[4.0, 5.0, 6.0]]), &device)
            .require_grad();
        let indices = Tensor::<TestAutodiffBackend, 2, Int>::from_data(
            TensorData::from([[2, 1, 0]]),
            &device,
        );

        let tensor_3 = tensor_1.clone().mul(tensor_2);
        let tensor_4 = tensor_3
            .clone()
            .scatter(1, indices, values.clone(), IndexingUpdateOp::Add);

        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = values.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_eq(&TensorData::from([[1., 2., 3., 4., 5., 6.]]), false);
        grad_2
            .to_data()
            .assert_eq(&TensorData::from([[1., 1., 1.]]), false);
    }
}
