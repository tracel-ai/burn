#[burn_tensor_testgen::testgen(memory_management)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Data, Tensor};

    #[test]
    fn test_mm_independant_trees() {
        let data = Data::from([[1.0, 2.0], [3.0, 4.0]]);
        let device = Default::default();

        // First tree
        let tensor_0 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_1 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_2 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_3 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

        let tensor_4 = tensor_0 * tensor_1;
        let tensor_5 = tensor_2 * tensor_3;
        let tensor_6 = tensor_4 * tensor_5;

        // Second tree
        let tensor_7 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_8 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_9 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_10 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

        let tensor_11 = tensor_7.clone() * tensor_8.clone();
        let tensor_12 = tensor_9.clone() * tensor_10.clone();
        let tensor_13 = tensor_11 * tensor_12;

        let grads = tensor_6.backward();
        let grads = tensor_13.backward();

        assert!(tensor_7.grad(&grads).is_some());
        assert!(tensor_8.grad(&grads).is_some());
        assert!(tensor_9.grad(&grads).is_some());
        assert!(tensor_10.grad(&grads).is_some());
    }

    #[test]
    #[should_panic]
    fn test_mm_crossover_trees_root_unavailable() {
        let data = Data::from([[1.0, 2.0], [3.0, 4.0]]);
        let device = Default::default();

        // First tree
        let tensor_0 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_1 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_2 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_3 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

        let tensor_4 = tensor_0 * tensor_1;
        let tensor_5 = tensor_2 * tensor_3;
        let tensor_6 = tensor_4.clone() * tensor_5;

        // Second tree
        let tensor_7 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_8 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

        let tensor_9 = tensor_7.clone() * tensor_8.clone();
        let tensor_10 = tensor_4 * tensor_9;

        let grads = tensor_6.backward();
        let grads = tensor_10.backward();
    }

    #[test]
    fn test_mm_crossover_trees_with_referred_subtree() {
        let data = Data::from([[1.0, 2.0], [3.0, 4.0]]);
        let device = Default::default();

        // First tree
        let tensor_0 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_1 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_2 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_3 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

        let tensor_4 = tensor_0 * tensor_1;
        let tensor_5 = tensor_2 * tensor_3;
        let tensor_6 = tensor_4.clone() * tensor_5;

        // Second tree
        let tensor_7 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_8 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

        let tensor_9 = tensor_7.clone() * tensor_8.clone();
        let tensor_10 = tensor_4 * tensor_9.clone();

        let grads = tensor_6.backward();
        let grads = tensor_9.backward();
    }

    #[test]
    fn test_mm_three_crossover_trees_last_still_usable() {
        let data = Data::from([[1.0, 2.0], [3.0, 4.0]]);
        let device = Default::default();

        // First tree
        let tensor_0 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_1 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_2 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_3 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

        let tensor_4 = tensor_0 * tensor_1;
        let tensor_5 = tensor_2 * tensor_3;
        let tensor_6 = tensor_4 * tensor_5.clone();

        // Third tree
        let tensor_7 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_8 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_9 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_10 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

        let tensor_11 = tensor_7 * tensor_8;
        let tensor_12 = tensor_9 * tensor_10;
        let tensor_13 = tensor_11 * tensor_12.clone();

        // Second tree (in between)
        let tensor_14 = tensor_5 * tensor_12;

        let grads = tensor_6.backward();
        let grads = tensor_13.backward();
    }

    #[test]
    #[should_panic]
    fn test_mm_three_crossover_trees_middle_one_unavailable() {
        let data = Data::from([[1.0, 2.0], [3.0, 4.0]]);
        let device = Default::default();

        // First tree
        let tensor_0 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_1 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_2 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_3 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

        let tensor_4 = tensor_0 * tensor_1;
        let tensor_5 = tensor_2 * tensor_3;
        let tensor_6 = tensor_4 * tensor_5.clone();

        // Third tree
        let tensor_7 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_8 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_9 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_10 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

        let tensor_11 = tensor_7 * tensor_8;
        let tensor_12 = tensor_9 * tensor_10;
        let tensor_13 = tensor_11 * tensor_12.clone();

        // Second tree (in between)
        let tensor_14 = tensor_5 * tensor_12;

        let grads = tensor_6.backward();
        let grads = tensor_14.backward();
    }

    #[test]
    fn test_mm_self_referencing_tree() {
        let data = Data::from([[1.0, 2.0], [3.0, 4.0]]);
        let device = Default::default();

        // First tree
        let tensor_0 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_1 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
        let tensor_2 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

        let tensor_3 = tensor_0 * tensor_1;
        let tensor_5 = tensor_2 * tensor_3.clone();
        let tensor_6 = tensor_3 * tensor_5;

        let grads = tensor_6.backward();
    }

    #[test]
    fn test_mm_with_non_impacting_detach() {
        let data = Data::from([[1.0, 2.0], [3.0, 4.0]]);
        let device = Default::default();
        let tensor_1 =
            Tensor::<TestAutodiffBackend, 2>::from_data(data.clone(), &device).require_grad();
        let tensor_2 =
            Tensor::<TestAutodiffBackend, 2>::from_data(data.clone(), &device).require_grad();
        let tensor_3 = Tensor::<TestAutodiffBackend, 2>::from_data(data, &device).require_grad();

        let tensor_4 = tensor_1.clone() * tensor_2.clone();
        let tensor_5 = tensor_4.detach() * tensor_3.clone();

        let grads = tensor_5.backward();
        assert!(tensor_3.grad(&grads).is_some());
    }

    // #[test]
    fn test_mm_with_impacting_detach() {
        let data_1 = Data::from([[0.0, 1.0], [3.0, 4.0]]);
        let data_2 = Data::from([[6.0, 7.0], [9.0, 10.0]]);

        let device = Default::default();
        let tensor_1 = Tensor::<TestAutodiffBackend, 2>::from_data(data_1, &device).require_grad();
        let tensor_2 = Tensor::<TestAutodiffBackend, 2>::from_data(data_2, &device).require_grad();

        let tensor_3 = tensor_1.clone() * tensor_2.clone();

        let tensor_3a = tensor_3.clone() - tensor_3.detach().max_dim(1);
        let tensor_3b = tensor_3a.exp();
        let tensor_3c = tensor_3b.clone().sum_dim(1);

        let tensor_3d = tensor_3b.div(tensor_3c);

        let tensor_4 = tensor_3d.matmul(tensor_2.clone());

        let grads = tensor_4.backward();
        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[1.1797, 1.1797], [0.0055, 0.0055]]), 3);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[0.2534, 0.2862], [0.5286, 2.9317]]), 3);
    }

    #[test]
    fn test_mm_with_missing_require_grad() {
        let data = Data::from([[1.0, 2.0], [3.0, 4.0]]);
        let device = Default::default();
        let tensor_1 =
            Tensor::<TestAutodiffBackend, 2>::from_data(data.clone(), &device).require_grad();
        let tensor_2 = Tensor::<TestAutodiffBackend, 2>::from_data(data.clone(), &device);
        let tensor_3 = Tensor::<TestAutodiffBackend, 2>::from_data(data, &device);

        let tensor_4 = tensor_1.clone() * tensor_2.clone();
        let tensor_5 = tensor_4 * tensor_3.clone();

        let grads = tensor_5.backward();
        assert!(tensor_1.grad(&grads).is_some());
        assert!(tensor_2.grad(&grads).is_none());
        assert!(tensor_3.grad(&grads).is_none());
    }

    // #[test]
    // fn test_mm_with_detach() {
    //     let data = Data::from([[1.0, 2.0], [3.0, 4.0]]);
    //     let device = Default::default();
    //     let tensor_1 =
    //         Tensor::<TestAutodiffBackend, 2>::from_data(data.clone(), &device).require_grad();
    //     let tensor_2 =
    //         Tensor::<TestAutodiffBackend, 2>::from_data(data.clone(), &device).require_grad();
    //     let tensor_3 = Tensor::<TestAutodiffBackend, 2>::from_data(data, &device).require_grad();

    //     let tensor_4 = tensor_1.clone() * tensor_2.clone();
    //     let tensor_5 = tensor_4 * tensor_3.detach();

    //     let grads = tensor_5.backward();
    //     assert!(tensor_1.grad(&grads).is_some());
    //     assert!(tensor_2.grad(&grads).is_some());
    // }
}
