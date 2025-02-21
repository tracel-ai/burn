#[burn_tensor_testgen::testgen(ad_nonzero)]
mod tests {
    use super::*;
    use burn_tensor::{Bool, Int, Tensor, TensorData};

    #[test]
    fn should_diff_nonzero() {
        let data_1 = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
        let data_2 = TensorData::from([-1.0, 1.0]);
        let mask = TensorData::from([[false, true], [true, false]]);

        let device = Default::default();
        let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
        let tensor_2 = TestAutodiffTensor::<1>::from_data(data_2, &device).require_grad();

        // Multi-dimensional tensor indexing isn't really supported yet so the easiest way to do
        // this is to flatten the mask and tensor to get proper indexing. Anyway the returned tensor would
        // have dimensions different from the input, so this is somewhat equivalent.
        let mask =
            Tensor::<TestAutodiffBackend, 2, Bool>::from_bool(mask, &device).flatten::<1>(0, 1);
        let indices = mask.nonzero();
        let tensor_3 = tensor_1
            .clone()
            .flatten::<1>(0, 1)
            .select(0, indices[0].clone());

        // Vector dot product not supported (only 2D matmuls) so unsqueeze for test purposes
        let tensor_4 = tensor_2
            .clone()
            .unsqueeze_dim::<2>(0)
            .matmul(tensor_3.unsqueeze_dim(1));

        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_eq(&TensorData::from([[0.0, -1.0], [1.0, 0.0]]), false);
        grad_2
            .to_data()
            .assert_eq(&TensorData::from([2.0, 3.0]), false);
    }
}
