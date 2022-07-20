use crate::execute_ops;
use crate::{
    backend::autodiff::{ADFloat, ADFloatTensor, ADTensor},
    ops::{BinaryOps, BinaryOpsNodeState, BinaryRecordedOps},
    register_ops, TensorOpsMatmul,
};
use num_traits::Float;

register_ops!(
    ops BinaryOps<T, T, T>,
    name ADTensorMatmulOps,
    partial_left |state: &BinaryOpsNodeState<T, T, T>| {
        let out_grad = state.output.borrow().value().ones();
        let rhs = state.right.borrow().value().transpose();
        out_grad.matmul(&rhs)
    },
    partial_right |state: &BinaryOpsNodeState<T, T, T>| {
        let out_grad = state.output.borrow().value().ones();
        let lhs = state.left.borrow().value().transpose();
        lhs.matmul(&out_grad)
    },
);

impl<T, P, const D: usize> TensorOpsMatmul<P, D> for ADTensor<P, D, T>
where
    T: ADFloatTensor<P, D>,
    P: ADFloat,
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
    use crate::{backend::autodiff::helper::ADTchTensor, Data, TensorBase};

    #[test]
    fn should_diff_mul() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);

        let tensor_1 = ADTchTensor::from_data(data_1.clone());
        let tensor_2 = ADTchTensor::from_data(data_2.clone());

        let tensor_3 = &tensor_1.matmul(&tensor_2);
        tensor_3.backprob();

        let grad_1 = tensor_1.grad();
        let grad_2 = tensor_2.grad();

        assert_eq!(grad_1.into_data(), Data::from([[11.0, 5.0], [11.0, 5.0]]));
        assert_eq!(grad_2.into_data(), Data::from([[3.0, 3.0], [10.0, 10.0]]));
        assert_eq!(
            tensor_3.clone().into_data(),
            Data::from([[18.0, 28.0], [14.0, 23.0]])
        );
    }
}
