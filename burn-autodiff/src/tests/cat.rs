#[burn_tensor_testgen::testgen(ad_cat)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[test]
    fn should_diff_cat() {
        let data_1 = Data::<_, 2>::from([[2.0, -1.0], [5.0, 2.0]]);
        let data_2 = Data::<_, 2>::from([[5.0, 4.0], [-1.0, 4.0]]);

        let tensor_1 = TestADTensor::from_data(data_1);
        let tensor_2 = TestADTensor::from_data(data_2);

        let tensor_3 = tensor_1.matmul(&tensor_2);
        let grads = tensor_3.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        let mut tensor_1_list = Vec::new();
        let mut tensor_2_list = Vec::new();

        for i in 0..2 {
            tensor_1_list.push(TestADTensor::from_data(
                tensor_1.index([i..i + 1]).to_data(),
            ));
            tensor_2_list.push(TestADTensor::from_data(
                tensor_2.index([i..i + 1]).to_data(),
            ));
        }

        let tensor_1_cat = TestADTensor::cat(tensor_1_list.clone(), 0);
        let tensor_2_cat = TestADTensor::cat(tensor_2_list.clone(), 0);

        let tensor_3_cat = tensor_1_cat.matmul(&tensor_2_cat);
        let grads_cat = tensor_3_cat.backward();

        let grad_1_cat = tensor_1_cat.grad(&grads_cat).unwrap();
        let grad_2_cat = tensor_2_cat.grad(&grads_cat).unwrap();

        let grad_1_list_1 = tensor_1_list.get(0).unwrap().grad(&grads_cat).unwrap();
        let grad_1_list_2 = tensor_1_list.get(1).unwrap().grad(&grads_cat).unwrap();

        let grad_2_list_1 = tensor_2_list.get(0).unwrap().grad(&grads_cat).unwrap();
        let grad_2_list_2 = tensor_2_list.get(1).unwrap().grad(&grads_cat).unwrap();

        grad_1.to_data().assert_approx_eq(&grad_1_cat.to_data(), 3);
        grad_2.to_data().assert_approx_eq(&grad_2_cat.to_data(), 3);

        grad_1
            .index([0..1])
            .to_data()
            .assert_approx_eq(&grad_1_list_1.to_data(), 3);

        grad_1
            .index([1..2])
            .to_data()
            .assert_approx_eq(&grad_1_list_2.to_data(), 3);
        grad_2
            .index([0..1])
            .to_data()
            .assert_approx_eq(&grad_2_list_1.to_data(), 3);

        grad_2
            .index([1..2])
            .to_data()
            .assert_approx_eq(&grad_2_list_2.to_data(), 3);
    }
}
