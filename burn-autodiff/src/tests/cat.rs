#[burn_tensor_testgen::testgen(ad_cat)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Float};

    #[test]
    fn should_diff_cat() {
        let data_1 = Data::<_, 2>::from([[2.0, -1.0], [5.0, 2.0]]);
        let data_2 = Data::<_, 2>::from([[5.0, 4.0], [-1.0, 4.0]]);

        let tensor_1 = TestADTensor::from_data(data_1).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let grads = tensor_3.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        let mut tensor_1_list = Vec::new();
        let mut tensor_2_list = Vec::new();

        for i in 0..2 {
            tensor_1_list.push(tensor_1.clone().index([i..i + 1]).detach().require_grad());
            tensor_2_list.push(tensor_2.clone().index([i..i + 1]).detach().require_grad());
        }

        let tensor_1_cat = TestADTensor::cat(tensor_1_list.clone(), 0);
        let tensor_2_cat = TestADTensor::cat(tensor_2_list.clone(), 0);

        let tensor_3_cat = tensor_1_cat.clone().matmul(tensor_2_cat.clone());
        let grads_cat = tensor_3_cat.backward();

        let grad = |tensor: Option<&TestADTensor<2, Float>>| {
            tensor
                .map(|tensor| tensor.grad(&grads_cat).unwrap())
                .unwrap()
        };
        let grad_1_index_1 = grad(tensor_1_list.get(0));
        let grad_1_index_2 = grad(tensor_1_list.get(1));

        let grad_2_index_1 = grad(tensor_2_list.get(0));
        let grad_2_index_2 = grad(tensor_2_list.get(1));

        grad_1
            .clone()
            .index([0..1])
            .to_data()
            .assert_approx_eq(&grad_1_index_1.to_data(), 3);
        grad_1
            .index([1..2])
            .to_data()
            .assert_approx_eq(&grad_1_index_2.to_data(), 3);

        grad_2
            .clone()
            .index([0..1])
            .to_data()
            .assert_approx_eq(&grad_2_index_1.to_data(), 3);
        grad_2
            .index([1..2])
            .to_data()
            .assert_approx_eq(&grad_2_index_2.to_data(), 3);
    }
}
