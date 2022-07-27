use crate::graph::ops::{BinaryOps, BinaryOpsNodeState};
use crate::tensor::backend::autodiff::ADTensor;
use crate::tensor::ops::*;
use crate::tensor::{Element, Tensor};
use crate::{execute_ops, register_ops};

register_ops!(
    ops BinaryOps<T, T, T>,
    name ADTensorMatmulOps,
    partial_left |state: &BinaryOpsNodeState<T, T, T>| {
        let out_grad = state.output.grad();
        let rhs = state.right.value().transpose();
        out_grad.matmul(&rhs)
    },
    partial_right |state: &BinaryOpsNodeState<T, T, T>| {
        let out_grad = state.output.grad();
        let lhs = state.left.value().transpose();
        lhs.matmul(&out_grad)
    },
);

impl<T, P, const D: usize> TensorOpsMatmul<P, D> for ADTensor<P, D, T>
where
    T: Tensor<P, D>,
    P: Element,
{
    fn matmul(&self, other: &Self) -> Self {
        let node = execute_ops!(
            lhs self.node.clone(),
            rhs other.node.clone(),
            out TensorOpsMatmul::matmul(&self.tensor(), &other.tensor()),
            ops ADTensorMatmulOps::new(),
        );
        self.from_existing(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_matmul() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());

        let tensor_3 = &tensor_1.matmul(&tensor_2);
        let grads = tensor_3.backward();

        let grad_1 = grads.wrt(&tensor_1).unwrap();
        let grad_2 = grads.wrt(&tensor_2).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[11.0, 5.0], [11.0, 5.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[3.0, 3.0], [10.0, 10.0]]));
        assert_eq!(
            tensor_3.clone().into_data(),
            Data::from([[18.0, 28.0], [14.0, 23.0]])
        );
    }

    #[test]
    fn test_matmul_complex_1() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [13.0, -3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_3: Data<f64, 2> = Data::from([[2.0, 2.0], [2.0, 2.0]]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());
        let tensor_3 = TestADTensor::from_data(data_3.clone());

        let tensor_4 = tensor_1.matmul(&tensor_2);
        let tensor_5 = tensor_4.matmul(&tensor_3);

        let grads = tensor_5.backward();

        let grad_1 = grads.wrt(&tensor_1).unwrap();
        let grad_2 = grads.wrt(&tensor_2).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[44.0, 20.0], [44.0, 20.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[56.0, 56.0], [16.0, 16.0]]));
    }
    #[test]
    fn test_matmul_complex_2() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [13.0, -3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_3: Data<f64, 2> = Data::from([[2.0, 2.0], [2.0, 2.0]]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());
        let tensor_3 = TestADTensor::from_data(data_3.clone());

        let tensor_4 = tensor_1.matmul(&tensor_2);
        let tensor_5 = tensor_4.matmul(&tensor_3);
        let tensor_6 = tensor_1.matmul(&tensor_5);

        let grads = tensor_6.backward();

        let grad_1 = grads.wrt(&tensor_1).unwrap();
        let grad_2 = grads.wrt(&tensor_2).unwrap();

        assert_eq!(
            grad_1.to_data(),
            Data::from([[800.0, 792.0], [360.0, 592.0]])
        );
        assert_eq!(
            grad_2.to_data(),
            Data::from([[264., 264.0], [344.0, 344.0]])
        );
    }
}
